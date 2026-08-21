"""Direct Cartesian K-mode trajectory decoder used by the final model."""

from __future__ import annotations

import math

import torch
from torch import nn

class DirectTrajectoryDecoder(nn.Module):
    """Direct six-mode Cartesian decoder.

    This decoder deliberately does NOT require:
        route progress,
        route interpolation,
        bounded longitudinal corrections,
        or bounded lateral corrections.

    Each learned behavior mode attends directly to causal scene tokens and
    predicts a full future displacement sequence.

    DynamicsAnchor is used only as a soft initialization/base trajectory,
    not as a hard constraint.
    """

    def __init__(
        self,
        *,
        d_model: int,
        num_modes: int,
        future_steps: int,
        fps: int,
        use_anchor_calibration: bool = False,
        use_longitudinal_repair: bool = False,
        num_heads: int = 8,
        num_layers: int = 2,
    ) -> None:

        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads."
            )

        self.d_model = int(d_model)
        self.num_modes = int(num_modes)
        self.future_steps = int(future_steps)
        self.fps = int(fps)
        self.use_anchor_calibration = bool(
            use_anchor_calibration
        )

        self.use_longitudinal_repair = bool(
            use_longitudinal_repair
        )

        self.mode_embedding = nn.Parameter(
            torch.randn(
                num_modes,
                d_model,
            )
            * 0.02
        )

        self.ego_projection = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.LayerNorm(
                d_model,
            ),
            nn.GELU(),
        )

        self.horizon_projection = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.GELU(),
        )

        self.time_encoder = nn.Sequential(
            nn.Linear(
                4,
                d_model,
            ),
            nn.GELU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        # Current dynamics position x/y + one-step displacement x/y.
        self.dynamics_encoder = nn.Sequential(
            nn.Linear(
                4,
                d_model,
            ),
            nn.GELU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        # -------------------------------------------------------------
        # TEMPORAL LONGITUDINAL REPAIR
        #
        # Direct-K6 currently emits every future displacement from
        # time-indexed scene queries and then integrates them with cumsum.
        # Small systematic dx bias therefore accumulates strongly by 5 s.
        #
        # This optional module operates on the provisional Direct sequence
        # BEFORE the final cumsum.  It predicts only an additive dx
        # correction; dy is intentionally untouched.
        #
        # Zero initialization of the final layer makes this path exactly
        # identity-compatible with existing checkpoints.
        # -------------------------------------------------------------

        self.longitudinal_repair_gru = None
        self.longitudinal_repair_head = None

        if self.use_longitudinal_repair:

            self.longitudinal_repair_gru = nn.GRU(
                input_size=d_model + 8,
                hidden_size=d_model,
                num_layers=2,
                batch_first=True,
                dropout=0.0,
            )

            self.longitudinal_repair_head = nn.Sequential(
                nn.LayerNorm(
                    d_model,
                ),
                nn.Linear(
                    d_model,
                    d_model,
                ),
                nn.GELU(),
                nn.Linear(
                    d_model,
                    1,
                ),
            )

            nn.init.zeros_(
                self.longitudinal_repair_head[-1].weight
            )

            nn.init.zeros_(
                self.longitudinal_repair_head[-1].bias
            )

        # Explicit raw WZ polygon/sign tokens.
        self.wz_projection = nn.Sequential(
            nn.Linear(
                3,
                d_model,
            ),
            nn.GELU(),
            nn.Linear(
                d_model,
                d_model,
            ),
        )

        self.query_norm = nn.LayerNorm(
            d_model,
        )

        self.scene_norm = nn.LayerNorm(
            d_model,
        )

        self.cross_attention = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=d_model,
                    num_heads=num_heads,
                    dropout=0.10,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ]
        )

        self.cross_norm = nn.ModuleList(
            [
                nn.LayerNorm(
                    d_model,
                )
                for _ in range(num_layers)
            ]
        )

        self.ffn = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(
                        d_model,
                        4 * d_model,
                    ),
                    nn.GELU(),
                    nn.Dropout(
                        0.10,
                    ),
                    nn.Linear(
                        4 * d_model,
                        d_model,
                    ),
                )
                for _ in range(num_layers)
            ]
        )

        self.ffn_norm = nn.ModuleList(
            [
                nn.LayerNorm(
                    d_model,
                )
                for _ in range(num_layers)
            ]
        )

        # Unbounded Cartesian per-step residual over the dynamics prior.
        self.delta_head = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Linear(
                d_model,
                2,
            ),
        )

        # -------------------------------------------------------------
        # DIRECT ABSOLUTE ANCHOR CALIBRATION
        #
        # The base Direct-K6 decoder predicts per-step displacement
        # residuals and integrates them with cumsum. Small velocity bias can
        # therefore accumulate into large 3-5 s position error.
        #
        # This optional head predicts absolute Cartesian corrections at
        # 1 s / 3 s / 5 s from the corresponding direct hidden states.
        # Corrections are smoothly interpolated across the 25-step future.
        #
        # Zero final-layer initialization makes enabling this module exactly
        # equivalent to the existing checkpoint before calibration training.
        # -------------------------------------------------------------

        self.anchor_correction_head = None

        if self.use_anchor_calibration:
            self.anchor_correction_head = nn.Sequential(
                nn.Linear(
                    d_model,
                    d_model,
                ),
                nn.GELU(),
                nn.Linear(
                    d_model,
                    2,
                ),
            )

            nn.init.zeros_(
                self.anchor_correction_head[-1].weight
            )

            nn.init.zeros_(
                self.anchor_correction_head[-1].bias
            )

        # Fully causal mode confidence.
        self.mode_score_head = nn.Sequential(
            nn.Linear(
                d_model,
                d_model,
            ),
            nn.GELU(),
            nn.Linear(
                d_model,
                d_model // 2,
            ),
            nn.GELU(),
            nn.Linear(
                d_model // 2,
                1,
            ),
        )

        # Small but NONZERO initialization is intentional.
        #
        # Zero initialization makes all K modes exactly equal on step 1 and
        # gives WTA regression gradient only to mode 0.
        nn.init.normal_(
            self.delta_head[-1].weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.zeros_(
            self.delta_head[-1].bias,
        )

        nn.init.normal_(
            self.mode_score_head[-1].weight,
            mean=0.0,
            std=0.01,
        )

        nn.init.zeros_(
            self.mode_score_head[-1].bias,
        )

        future_time = (
            torch.arange(
                1,
                future_steps + 1,
                dtype=torch.float32,
            )
            /
            float(fps)
        )

        normalized_time = (
            future_time
            /
            future_time[-1].clamp_min(
                1.0e-6
            )
        )

        time_features = torch.stack(
            (
                normalized_time,
                normalized_time.square(),
                torch.sin(
                    math.pi
                    *
                    normalized_time
                ),
                torch.cos(
                    math.pi
                    *
                    normalized_time
                ),
            ),
            dim=-1,
        )

        horizon_index = torch.where(
            future_time <= 1.0,
            torch.zeros_like(
                future_time,
                dtype=torch.long,
            ),
            torch.where(
                future_time <= 3.0,
                torch.ones_like(
                    future_time,
                    dtype=torch.long,
                ),
                torch.full_like(
                    future_time,
                    2,
                    dtype=torch.long,
                ),
            ),
        )

        self.register_buffer(
            "direct_time_features",
            time_features,
            persistent=False,
        )

        self.register_buffer(
            "direct_horizon_index",
            horizon_index,
            persistent=False,
        )


        # -------------------------------------------------------------
        # Piecewise-linear interpolation weights for calibration anchors.
        #
        # Anchor indices at 5 Hz / 25 future steps:
        #   1 s -> index 4
        #   3 s -> index 14
        #   5 s -> index 24
        #
        # The implementation remains valid for the configured fps/future T.
        # -------------------------------------------------------------

        calibration_anchor_index = torch.tensor(
            [
                min(
                    self.fps - 1,
                    self.future_steps - 1,
                ),
                min(
                    3 * self.fps - 1,
                    self.future_steps - 1,
                ),
                self.future_steps - 1,
            ],
            dtype=torch.long,
        )

        calibration_weights = torch.zeros(
            self.future_steps,
            3,
            dtype=torch.float32,
        )

        final_time_s = (
            float(self.future_steps)
            /
            float(self.fps)
        )

        for step in range(self.future_steps):
            t = (
                float(step + 1)
                /
                float(self.fps)
            )

            if t <= 1.0:
                # Origin correction is exactly zero.
                calibration_weights[
                    step,
                    0,
                ] = t

            elif t <= 3.0:
                alpha = (
                    (t - 1.0)
                    /
                    2.0
                )

                calibration_weights[
                    step,
                    0,
                ] = 1.0 - alpha

                calibration_weights[
                    step,
                    1,
                ] = alpha

            else:
                denominator = max(
                    final_time_s - 3.0,
                    1.0e-6,
                )

                alpha = min(
                    max(
                        (t - 3.0)
                        /
                        denominator,
                        0.0,
                    ),
                    1.0,
                )

                calibration_weights[
                    step,
                    1,
                ] = 1.0 - alpha

                calibration_weights[
                    step,
                    2,
                ] = alpha

        self.register_buffer(
            "direct_calibration_anchor_index",
            calibration_anchor_index,
            persistent=False,
        )

        self.register_buffer(
            "direct_calibration_weights",
            calibration_weights,
            persistent=False,
        )

    def forward(
        self,
        *,
        ego_context: torch.Tensor,
        horizon_context: torch.Tensor,
        lane_states: torch.Tensor,
        lane_mask: torch.Tensor,
        agent_states: torch.Tensor,
        agent_mask: torch.Tensor,
        worker_tokens: torch.Tensor,
        worker_mask: torch.Tensor,
        wz_feat: torch.Tensor,
        dynamics_xy: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        batch_size = ego_context.shape[0]

        if ego_context.shape != (
            batch_size,
            self.d_model,
        ):
            raise ValueError(
                "ego_context has wrong shape."
            )

        if horizon_context.shape != (
            batch_size,
            3,
            self.d_model,
        ):
            raise ValueError(
                "horizon_context must have shape [B,3,D]."
            )

        if dynamics_xy.shape != (
            batch_size,
            self.future_steps,
            2,
        ):
            raise ValueError(
                "dynamics_xy must have shape [B,T,2]."
            )

        # -------------------------------------------------------------
        # Explicit scene-token bank.
        #
        # Ego token is always valid so MHA can never receive a row where
        # every key is padding.
        # -------------------------------------------------------------

        ego_token = ego_context[
            :,
            None,
        ]

        ego_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=ego_context.device,
        )

        wz_feat = wz_feat.to(
            ego_context.dtype
        )

        wz_tokens = self.wz_projection(
            wz_feat
        )

        wz_mask = (
            wz_feat[
                ...,
                2
            ]
            >
            0
        )

        scene_tokens = torch.cat(
            (
                ego_token,
                lane_states,
                agent_states,
                worker_tokens,
                wz_tokens,
            ),
            dim=1,
        )

        scene_mask = torch.cat(
            (
                ego_mask,
                lane_mask.bool(),
                agent_mask.bool(),
                worker_mask.bool(),
                wz_mask.bool(),
            ),
            dim=1,
        )

        scene_tokens = self.scene_norm(
            scene_tokens
        )

        # -------------------------------------------------------------
        # Dynamics features are INPUT CONTEXT, not a constraint.
        # -------------------------------------------------------------

        origin = torch.zeros(
            batch_size,
            1,
            2,
            dtype=dynamics_xy.dtype,
            device=dynamics_xy.device,
        )

        previous = torch.cat(
            (
                origin,
                dynamics_xy[
                    :,
                    :-1,
                ],
            ),
            dim=1,
        )

        dynamics_delta = (
            dynamics_xy
            -
            previous
        )

        dynamics_features = torch.cat(
            (
                dynamics_xy / 50.0,
                (
                    dynamics_delta
                    *
                    float(
                        self.fps
                    )
                )
                /
                20.0,
            ),
            dim=-1,
        )

        dynamics_context = self.dynamics_encoder(
            dynamics_features
        )

        # -------------------------------------------------------------
        # K learned behavioral hypotheses x T future queries.
        # -------------------------------------------------------------

        time_context = self.time_encoder(
            self.direct_time_features.to(
                dtype=ego_context.dtype,
                device=ego_context.device,
            )
        )

        horizon_context_t = (
            horizon_context.index_select(
                1,
                self.direct_horizon_index.to(
                    device=ego_context.device,
                ),
            )
        )

        ego_context_q = self.ego_projection(
            ego_context
        )

        mode = self.mode_embedding.to(
            dtype=ego_context.dtype
        )

        query = (
            ego_context_q[
                :,
                None,
                None,
            ]
            +
            mode[
                None,
                :,
                None,
            ]
            +
            time_context[
                None,
                None,
            ]
            +
            self.horizon_projection(
                horizon_context_t
            )[
                :,
                None,
            ]
            +
            dynamics_context[
                :,
                None,
            ]
        )

        query = self.query_norm(
            query
        )

        query = query.reshape(
            batch_size,
            self.num_modes
            *
            self.future_steps,
            self.d_model,
        )

        key_padding_mask = (
            ~scene_mask
        )

        for (
            attention,
            attention_norm,
            ffn,
            ffn_norm,
        ) in zip(
            self.cross_attention,
            self.cross_norm,
            self.ffn,
            self.ffn_norm,
        ):

            attended, _ = attention(
                query=query,
                key=scene_tokens,
                value=scene_tokens,
                key_padding_mask=key_padding_mask,
                need_weights=False,
            )

            query = attention_norm(
                query
                +
                attended
            )

            query = ffn_norm(
                query
                +
                ffn(
                    query
                )
            )

        hidden = query.reshape(
            batch_size,
            self.num_modes,
            self.future_steps,
            self.d_model,
        )

        residual_delta = self.delta_head(
            hidden
        )

        # Dynamics is merely the starting point. Residual delta is unbounded.
        pre_repair_delta = (
            dynamics_delta[
                :,
                None,
            ]
            +
            residual_delta
        )

        # Provisional integrated prediction used as explicit temporal state.
        pre_repair_pred_xy = torch.cumsum(
            pre_repair_delta,
            dim=2,
        )

        longitudinal_repair_delta = torch.zeros_like(
            pre_repair_delta
        )

        if self.use_longitudinal_repair:

            if (
                self.longitudinal_repair_gru is None
                or
                self.longitudinal_repair_head is None
            ):
                raise RuntimeError(
                    "Longitudinal repair enabled but module is missing."
                )

            dynamics_xy_modes = dynamics_xy[
                :,
                None,
            ].expand(
                -1,
                self.num_modes,
                -1,
                -1,
            )

            dynamics_delta_modes = dynamics_delta[
                :,
                None,
            ].expand(
                -1,
                self.num_modes,
                -1,
                -1,
            )

            repair_features = torch.cat(
                (
                    hidden,

                    # Absolute progression state.
                    pre_repair_pred_xy / 50.0,
                    dynamics_xy_modes / 50.0,

                    # Velocity-like displacement state.
                    (
                        pre_repair_delta
                        *
                        float(self.fps)
                        /
                        20.0
                    ),
                    (
                        dynamics_delta_modes
                        *
                        float(self.fps)
                        /
                        20.0
                    ),
                ),
                dim=-1,
            )

            repair_sequence, _ = (
                self.longitudinal_repair_gru(
                    repair_features.reshape(
                        batch_size
                        *
                        self.num_modes,
                        self.future_steps,
                        self.d_model + 8,
                    )
                )
            )

            repair_sequence = repair_sequence.reshape(
                batch_size,
                self.num_modes,
                self.future_steps,
                self.d_model,
            )

            raw_longitudinal_repair = (
                self.longitudinal_repair_head(
                    repair_sequence
                )
            )

            # Maximum |dx| repair is 0.50 m per 0.2 s step.
            # This is ample for the observed 3-5 s drift while preventing
            # pathological corrections during early training.
            longitudinal_repair_x = (
                0.50
                *
                torch.tanh(
                    raw_longitudinal_repair
                )
            )

            longitudinal_repair_delta = torch.cat(
                (
                    longitudinal_repair_x,
                    torch.zeros_like(
                        longitudinal_repair_x
                    ),
                ),
                dim=-1,
            )

        delta = (
            pre_repair_delta
            +
            longitudinal_repair_delta
        )

        base_pred_xy = torch.cumsum(
            delta,
            dim=2,
        )

        pred_xy = base_pred_xy

        if self.use_anchor_calibration:
            if self.anchor_correction_head is None:
                raise RuntimeError(
                    "Anchor calibration enabled but head is missing."
                )

            calibration_anchor_hidden = hidden.index_select(
                2,
                self.direct_calibration_anchor_index.to(
                    device=hidden.device,
                ),
            )

            # [B,K,3,2]
            calibration_anchor_delta = (
                self.anchor_correction_head(
                    calibration_anchor_hidden
                )
            )

            interpolation_weights = (
                self.direct_calibration_weights.to(
                    dtype=calibration_anchor_delta.dtype,
                    device=calibration_anchor_delta.device,
                )
            )

            # [T,3] x [B,K,3,2] -> [B,K,T,2]
            calibration_xy = torch.einsum(
                "th,bkhd->bktd",
                interpolation_weights,
                calibration_anchor_delta,
            )

            pred_xy = (
                base_pred_xy
                +
                calibration_xy
            )

        # Endpoint gets extra weight in mode representation because our
        # deployment target is explicitly strict on FDE.
        mode_hidden = (
            0.50
            *
            hidden.mean(
                dim=2
            )
            +
            0.50
            *
            hidden[
                :,
                :,
                -1,
            ]
        )

        mode_logits = self.mode_score_head(
            mode_hidden
        ).squeeze(
            -1
        )

        mode_prob = torch.softmax(
            mode_logits.float(),
            dim=-1,
        ).to(
            mode_logits.dtype
        )

        return {
            "pred_xy": pred_xy,
            "direct_base_pred_xy": base_pred_xy,
            "mode_logits": mode_logits,
            "mode_prob": mode_prob,
            "direct_delta": delta,
            "direct_residual_delta": residual_delta,
            "direct_pre_repair_delta": pre_repair_delta,
            "direct_pre_repair_pred_xy": pre_repair_pred_xy,
            "direct_longitudinal_repair_delta": longitudinal_repair_delta,
            "direct_hidden": hidden,
        }
