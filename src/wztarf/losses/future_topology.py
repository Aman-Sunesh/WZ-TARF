"""Future-path supervision for temporary topology and differentiable route set."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wztarf.data.future_topology_targets import (
    FutureTopologyTargets,
)


@dataclass(frozen=True)
class FutureTopologyLoss:
    total: torch.Tensor
    node: torch.Tensor
    edge: torch.Tensor
    transition: torch.Tensor


def _positive_probability_loss(
    probability: torch.Tensor,
    positive_mask: torch.Tensor,
) -> torch.Tensor:
    """Positive-only Bernoulli NLL with recoverable saturated gradients.

    Do NOT clamp probabilities to 1e-6 here.  A hard lower clamp makes
    every positive prediction below that threshold receive zero gradient,
    permanently trapping a collapsed topology head.

    The tiny additive floor exists only to keep log(0) finite.  For any
    representable sigmoid probability above float32 tiny, the gradient
    remains effectively the stable Bernoulli gradient with respect to
    the upstream logit.
    """

    probability = probability.float()
    positive_mask = positive_mask.bool()

    weight = positive_mask.to(
        probability.dtype
    )

    tiny = torch.finfo(
        probability.dtype
    ).tiny

    positive_nll = -torch.log(
        probability + tiny
    )

    return (
        positive_nll
        *
        weight
    ).sum() / weight.sum().clamp_min(
        1.0
    )



def _negative_probability_loss(
    probability: torch.Tensor,
    negative_mask: torch.Tensor,
) -> torch.Tensor:
    """Negative-only Bernoulli NLL without a dead hard clamp."""

    probability = probability.float()
    negative_mask = negative_mask.bool()

    weight = negative_mask.to(
        probability.dtype
    )

    tiny = torch.finfo(
        probability.dtype
    ).tiny

    negative_nll = -torch.log(
        (
            1.0
            -
            probability
        )
        +
        tiny
    )

    return (
        negative_nll
        *
        weight
    ).sum() / weight.sum().clamp_min(
        1.0
    )



def future_topology_supervision_loss(
    *,
    node_viability: torch.Tensor,
    edge_viability: torch.Tensor,
    route_edge_occupancy: torch.Tensor,
    goal_prob: torch.Tensor,
    targets: FutureTopologyTargets,
    blocked_edge_compatibility: torch.Tensor | None = None,
    blocked_edge_mask: torch.Tensor | None = None,
    blocked_negative_weight: float = 0.25,
) -> FutureTopologyLoss:
    """Supervise temporary topology without destroying alternative routes.

    Future path:
        proves traversed nodes/edges are viable.

    WZ geometric compatibility:
        provides conservative blocked-edge negative evidence.

    Unvisited-but-not-blocked graph elements:
        remain unlabeled.

    Route-set transition objective:
        at least one of K soft route hypotheses must cover each realized
        graph transition.

    MAP_EXIT:
        if the future leaves the represented graph, at least one route
        hypothesis must allocate probability to MAP_EXIT.
    """

    if node_viability.shape != (
        targets.node_positive.shape
    ):
        raise ValueError(
            "node_viability/target shape mismatch."
        )

    if edge_viability.shape != (
        targets.edge_positive.shape
    ):
        raise ValueError(
            "edge_viability/target shape mismatch."
        )

    if (
        route_edge_occupancy.ndim != 3
        or
        route_edge_occupancy.shape[
            0
        ]
        !=
        edge_viability.shape[0]
        or
        route_edge_occupancy.shape[
            2
        ]
        !=
        edge_viability.shape[1]
    ):
        raise ValueError(
            "route_edge_occupancy must have shape [B,K,E]."
        )

    # ----------------------------------------------------------
    # L_node:
    # realized-path nodes must remain viable.
    # ----------------------------------------------------------

    node_loss = _positive_probability_loss(
        node_viability,
        targets.node_positive,
    )

    # ----------------------------------------------------------
    # L_edge:
    # positive realized edges + conservative WZ-blocked negatives.
    #
    # Positives and negatives are averaged separately so hundreds
    # of negatives cannot overwhelm the small positive set.
    # ----------------------------------------------------------

    edge_positive_loss = (
        _positive_probability_loss(
            edge_viability,
            targets.edge_positive,
        )
    )

    if (
        blocked_edge_compatibility
        is not None
        and
        blocked_edge_mask
        is not None
    ):
        if (
            blocked_edge_compatibility.shape
            !=
            edge_viability.shape
        ):
            raise ValueError(
                "blocked_edge_compatibility has wrong shape."
            )

        if (
            blocked_edge_mask.shape
            !=
            edge_viability.shape
        ):
            raise ValueError(
                "blocked_edge_mask has wrong shape."
            )

        blocked = (
            blocked_edge_mask.bool()
            &
            (
                blocked_edge_compatibility
                <
                0.5
            )
            &
            ~targets.edge_positive
        )

        edge_negative_loss = (
            _negative_probability_loss(
                edge_viability,
                blocked,
            )
        )

        edge_loss = (
            edge_positive_loss
            +
            float(
                blocked_negative_weight
            )
            *
            edge_negative_loss
        )

    else:
        edge_loss = edge_positive_loss

    # ----------------------------------------------------------
    # L_transition:
    #
    # Independent per-edge membership allows one route to contain
    # several graph edges. Compute differentiable "set union"
    # probability:
    #
    # P(edge covered by any route)
    #     = 1 - product_k (1 - p_k)
    #
    # This is smooth and sends gradients to every plausible route.
    # ----------------------------------------------------------

    route_edge_probability = (
        route_edge_occupancy
        .float()
        .clamp(
            1.0e-6,
            1.0 - 1.0e-6,
        )
    )

    log_not_covered = torch.log1p(
        -route_edge_probability
    ).sum(
        dim=1
    )

    edge_covered = (
        1.0
        -
        torch.exp(
            log_not_covered
        )
    ).clamp(
        1.0e-6,
        1.0,
    )

    transition_weight = (
        targets.edge_positive.to(
            edge_covered.dtype
        )
    )

    transition_edge_loss = (
        -torch.log(
            edge_covered
        )
        *
        transition_weight
    ).sum() / transition_weight.sum().clamp_min(
        1.0
    )

    # ----------------------------------------------------------
    # MAP_EXIT is positive-only as well.
    # An in-map realized future does NOT prove that MAP_EXIT was
    # impossible as an alternative.
    # ----------------------------------------------------------

    if (
        goal_prob.ndim != 3
        or
        goal_prob.shape[0]
        !=
        node_viability.shape[0]
    ):
        raise ValueError(
            "goal_prob must have shape [B,K,C]."
        )

    exit_probability = (
        goal_prob[
            ...,
            -1
        ]
        .float()
        .clamp(
            1.0e-6,
            1.0 - 1.0e-6,
        )
    )

    exit_not_covered = torch.log1p(
        -exit_probability
    ).sum(
        dim=1
    )

    exit_covered = (
        1.0
        -
        torch.exp(
            exit_not_covered
        )
    ).clamp(
        1.0e-6,
        1.0,
    )

    map_exit_weight = (
        targets.map_exit.to(
            exit_covered.dtype
        )
    )

    map_exit_loss = (
        -torch.log(
            exit_covered
        )
        *
        map_exit_weight
    ).sum() / map_exit_weight.sum().clamp_min(
        1.0
    )

    transition_loss = (
        transition_edge_loss
        +
        map_exit_loss
    )

    total = (
        node_loss
        +
        edge_loss
        +
        transition_loss
    )

    return FutureTopologyLoss(
        total=total,
        node=node_loss,
        edge=edge_loss,
        transition=transition_loss,
    )
