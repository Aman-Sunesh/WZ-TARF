"""Encode sparse surrounding-agent histories and pool relevant agents."""

from __future__ import annotations

import torch
from torch import nn


class AgentEncoder(nn.Module):
    """Apply one shared temporal encoder to all surrounding agents.

    Performance note:
        Phase-B forecasting only needs each agent's last encoded state.  The
        original implementation projected every GRU timestep to ``output_dim``
        and then kept one timestep, and recovered last XY with nested Python
        loops.  This implementation gathers the last GRU state first, projects
        only what is needed, and vectorizes last-XY extraction.  Full temporal
        projected states remain available for Phase-A pretraining through
        ``return_temporal=True``.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.projection = nn.Linear(hidden_dim, output_dim)
        self.relevance = nn.Sequential(
            nn.Linear(2 * output_dim, output_dim),
            nn.ReLU(),
            nn.Linear(
                output_dim,
                1,
                bias=False,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        ego_context: torch.Tensor,
        *,
        return_temporal: bool = True,
    ) -> dict[str, torch.Tensor | None]:
        """Encode ``[B,T,A,F]`` histories and relevance-pool valid agents."""
        if x.ndim != 4:
            raise ValueError("Agent history must have shape [B, T, A, F].")

        batch_size, history_steps, num_agents, feature_dim = x.shape
        if mask.shape != (batch_size, history_steps, num_agents):
            raise ValueError("Agent mask must have shape [B, T, A].")
        if ego_context.shape != (batch_size, self.output_dim):
            raise ValueError("ego_context must have shape [B, D].")

        mask = mask.bool()

        # [B,A,T,F] -> [B*A,T,F]
        agent_sequence = x.permute(0, 2, 1, 3).reshape(
            batch_size * num_agents,
            history_steps,
            feature_dim,
        )
        flat_mask = mask.permute(0, 2, 1).reshape(
            batch_size * num_agents,
            history_steps,
        )
        agent_sequence = agent_sequence * flat_mask[..., None].to(
            agent_sequence.dtype
        )

        temporal, _ = self.gru(agent_sequence)

        time_index = torch.arange(history_steps, device=x.device)[None]
        last_index = (flat_mask.long() * time_index).amax(dim=1)
        flat_index = torch.arange(temporal.shape[0], device=x.device)
        present = flat_mask.any(dim=1)

        # Project only the state consumed by Phase B.  This avoids an otherwise
        # unnecessary [B*A*T,H] -> [B*A*T,D] projection in supervised runs.
        raw_last_state = temporal[flat_index, last_index]
        last_state = self.projection(raw_last_state)
        last_state = last_state * present[:, None].to(last_state.dtype)
        agent_states = last_state.reshape(batch_size, num_agents, self.output_dim)

        agent_temporal_states: torch.Tensor | None = None
        if return_temporal:
            agent_temporal_states = self.projection(temporal).reshape(
                batch_size,
                num_agents,
                history_steps,
                self.output_dim,
            ).permute(0, 2, 1, 3).contiguous()

        agent_present = present.reshape(batch_size, num_agents)
        ego = ego_context[:, None].expand(-1, num_agents, -1)
        relevance_logits = self.relevance(
            torch.cat((agent_states, ego), dim=-1)
        ).squeeze(-1)
        relevance_logits = relevance_logits.masked_fill(
            ~agent_present,
            torch.finfo(relevance_logits.dtype).min,
        )

        weights = torch.softmax(relevance_logits, dim=1)
        weights = weights * agent_present.to(weights.dtype)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        pooled = (weights[..., None] * agent_states).sum(dim=1)

        # Vectorized last-valid XY.  ``last_index`` already contains the last
        # valid temporal index for every flattened agent.
        raw_xy = x[..., :2].permute(0, 2, 1, 3)  # [B,A,T,2]
        last_index_ba = last_index.reshape(batch_size, num_agents)
        gather_index = last_index_ba[..., None, None].expand(-1, -1, 1, 2)
        last_xy = raw_xy.gather(2, gather_index).squeeze(2)
        last_xy = last_xy * agent_present[..., None].to(last_xy.dtype)

        return {
            "agent_temporal_states": agent_temporal_states,
            "agent_states": agent_states,
            "agent_context": pooled,
            "agent_relevance": weights,
            "agent_mask": agent_present,
            "agent_xy": last_xy,
        }
