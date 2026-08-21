"""Construct lane, MAP_EXIT, longitudinal-goal, and road-reliability targets."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from wztarf.data.map_coverage import (
    build_map_coverage_mask_batched,
    distance_to_lane_union_batched,
    distance_to_lanes_batched,
    selected_lane_longitudinal_offset_batched,
)


@dataclass
class SupervisedTargets:
    """Training targets derived from GT future and represented map geometry."""

    goal_target: torch.Tensor
    goal_valid: torch.Tensor
    goal_offset_target: torch.Tensor
    lane_goal_mask: torch.Tensor
    road_reliability_mask: torch.Tensor


def build_supervised_targets(
    *,
    gt_xy: torch.Tensor,
    lane_feat: torch.Tensor,
    lane_point_mask: torch.Tensor,
    lane_mask: torch.Tensor,
    retained_lane_mask: torch.Tensor,
    association_tolerance_m: float = 0.25,
    road_gt_tolerance_m: float = 0.25,
    compute_road_reliability: bool = True,
) -> SupervisedTargets:
    """Build supervised map targets with batched accelerator geometry."""
    if gt_xy.ndim != 3 or gt_xy.shape[-1] != 2:
        raise ValueError("gt_xy must have shape [B, T, 2].")

    batch_size, future_steps, _ = gt_xy.shape
    num_lanes = lane_feat.shape[1]

    if retained_lane_mask.shape != (batch_size, num_lanes):
        raise ValueError(
            "retained_lane_mask must have shape [B, L]."
        )

    raw_lane_mask = lane_mask.bool()

    coverage = build_map_coverage_mask_batched(
        gt_xy,
        lane_feat,
        lane_point_mask,
        raw_lane_mask,
    )

    if compute_road_reliability:
        road_distance = distance_to_lane_union_batched(
            gt_xy,
            lane_feat,
            lane_point_mask,
            raw_lane_mask,
        )

        road_reliability = (
            coverage
            &
            (road_distance <= road_gt_tolerance_m)
        )
    else:
        road_reliability = torch.zeros(
            batch_size,
            future_steps,
            dtype=torch.bool,
            device=gt_xy.device,
        )

    endpoint = gt_xy[:, -1]
    endpoint_covered = coverage[:, -1]
    map_exit = ~endpoint_covered

    endpoint_lane_distance = distance_to_lanes_batched(
        endpoint[:, None, :],
        lane_feat,
        lane_point_mask,
        raw_lane_mask,
    ).squeeze(-1)

    retained = (
        retained_lane_mask.bool()
        &
        raw_lane_mask
    )

    retained_distance = torch.where(
        retained,
        endpoint_lane_distance,
        torch.full_like(
            endpoint_lane_distance,
            float("inf"),
        ),
    )

    best_distance, best_lane = retained_distance.min(dim=1)

    associated = (
        endpoint_covered
        &
        torch.isfinite(best_distance)
        &
        (best_distance <= association_tolerance_m)
    )

    map_exit_class = num_lanes

    goal_target = torch.zeros(
        batch_size,
        dtype=torch.long,
        device=gt_xy.device,
    )

    goal_target = torch.where(
        map_exit,
        torch.full_like(goal_target, map_exit_class),
        goal_target,
    )

    goal_target = torch.where(
        associated,
        best_lane.long(),
        goal_target,
    )

    goal_valid = map_exit | associated
    lane_goal_mask = associated

    selected_offset = selected_lane_longitudinal_offset_batched(
        endpoint,
        best_lane,
        lane_feat,
        lane_point_mask,
    )

    goal_offset_target = torch.where(
        associated,
        selected_offset,
        torch.zeros_like(selected_offset),
    )

    return SupervisedTargets(
        goal_target=goal_target,
        goal_valid=goal_valid,
        goal_offset_target=goal_offset_target,
        lane_goal_mask=lane_goal_mask,
        road_reliability_mask=road_reliability,
    )
