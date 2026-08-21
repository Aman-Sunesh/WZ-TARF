from __future__ import annotations

import torch

from wztarf.pipeline.common import build_causal_state, canonical_aux_weights
from wztarf.pipeline.late import (
    EndpointZeroCalibrator,
    apply_fixed12,
    endpoint_zero_basis,
)
from wztarf.postprocess.action_policy import gate_features


def synthetic_batch(batch_size: int = 3) -> dict[str, torch.Tensor]:
    return {
        "ego_hist": torch.randn(batch_size, 10, 6),
        "control_hist": torch.randn(batch_size, 10, 3),
        "gaze_feat": torch.randn(batch_size, 10, 3),
        "wz_worker_feat": torch.randn(batch_size, 2, 3),
    }


def test_causal_state_is_exactly_78d_and_ignores_explicit_wz_feat() -> None:
    batch = synthetic_batch()
    batch["wz_feat"] = torch.randn(3, 5, 3)
    first = build_causal_state(batch)
    batch["wz_feat"] = torch.randn(3, 5, 3) * 1000
    second = build_causal_state(batch)
    assert first.shape == (3, 78)
    torch.testing.assert_close(first, second)


def test_endpoint_zero_basis_is_zero_at_final_step() -> None:
    basis = endpoint_zero_basis(7, 25)
    assert basis.shape == (7, 25)
    torch.testing.assert_close(basis[:, -1], torch.zeros(7), atol=0.0, rtol=0.0)


def test_endpoint_zero_calibrator_matches_historical_parameter_count() -> None:
    model = EndpointZeroCalibrator(n_basis=7, cap=1.5)
    assert sum(p.numel() for p in model.parameters()) == 12745


def test_a20_feature_dimension_is_114() -> None:
    pred = torch.randn(4, 6, 25, 2)
    state = torch.randn(4, 78)
    feature, far_idx = gate_features(pred, state)
    assert feature.shape == (4, 114)
    assert far_idx.shape == (4,)


def test_fixed12_is_x_only() -> None:
    pred = torch.randn(2, 6, 25, 2)
    a = torch.randn(6)
    b = torch.randn(6)
    out = apply_fixed12(pred, a, b)
    torch.testing.assert_close(out[..., 1], pred[..., 1])


def test_no_wz_late_auxiliary_budget_preserves_ablation() -> None:
    wz = {"model": {"topology_mode": "workzone", "use_workers": True}}
    no_wz = {"model": {"topology_mode": "static", "use_workers": True}}
    wz_weights = canonical_aux_weights(wz)
    no_wz_weights = canonical_aux_weights(no_wz)
    assert wz_weights["topology"] > 0
    assert wz_weights["wz_geometry"] > 0
    assert no_wz_weights["topology"] == 0
    assert no_wz_weights["wz_geometry"] == 0
    assert no_wz_weights["worker"] == wz_weights["worker"] > 0


def test_native_k64_wrapper_expands_native_direct_decoder() -> None:
    from tests.models.test_wztarf_forward import _batch
    from wztarf.models import WZTARF, WZTARFConfig
    from wztarf.pipeline.direct import NativeK64Distribution

    batch = _batch()
    model = WZTARF(
        WZTARFConfig(
            d_model=32,
            motion_hidden=32,
            control_hidden=16,
            gaze_hidden=16,
            agent_hidden=16,
            num_modes=6,
            num_edge_types=4,
            use_direct_decoder=True,
            use_direct_anchor_calibration=True,
            use_direct_longitudinal_repair=False,
            aux_dropout_controls=0.0,
            aux_dropout_gaze=0.0,
            aux_dropout_workers=0.0,
        )
    )
    captured = {}

    def hook(module, positional, kwargs):
        captured["kwargs"] = kwargs

    handle = model.direct_trajectory_decoder.register_forward_pre_hook(hook, with_kwargs=True)
    model.eval()
    with torch.no_grad():
        _ = model(batch)
    handle.remove()
    torch.manual_seed(123)
    distribution = NativeK64Distribution(model.direct_trajectory_decoder)
    with torch.no_grad():
        output = distribution(captured["kwargs"])
    assert output["pred_xy"].shape == (2, 64, 25, 2)
    assert output["logits"].shape == (2, 64)
    assert output["sigma_s"].shape == (2, 64)
    assert output["sigma_d"].shape == (2, 64)


def test_late_train_mode_zeroes_dropout_without_disabling_gru_backward() -> None:
    from torch import nn
    from wztarf.models import WZTARF, WZTARFConfig
    from wztarf.pipeline.direct import _zero_training_dropout

    model = WZTARF(
        WZTARFConfig(
            d_model=32, motion_hidden=32, control_hidden=16, gaze_hidden=16,
            agent_hidden=16, num_modes=6, num_edge_types=4,
            use_direct_decoder=True, use_direct_anchor_calibration=True,
            use_direct_longitudinal_repair=True,
            aux_dropout_controls=0.0, aux_dropout_gaze=0.0, aux_dropout_workers=0.0,
        )
    )
    model.train()
    _zero_training_dropout(model)
    decoder = model.direct_trajectory_decoder
    assert decoder is not None
    assert decoder.training
    assert decoder.longitudinal_repair_gru is not None
    assert decoder.longitudinal_repair_gru.training
    assert all(m.p == 0.0 for m in model.modules() if isinstance(m, nn.Dropout))
    assert all(m.dropout == 0.0 for m in model.modules() if isinstance(m, nn.MultiheadAttention))


def test_recovered_headonly_target_anchor_k64_and_a3_recipe_is_locked() -> None:
    from pathlib import Path
    import yaml

    root = Path(__file__).resolve().parents[1]
    for name in ("wz.yaml", "no_wz.yaml"):
        cfg = yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8"))
        head = cfg["canonical_pipeline"]["dense_progress_headonly"]
        target = cfg["canonical_pipeline"]["direct_target"]
        anchor = cfg["canonical_pipeline"]["anchor_calibration"]
        k64 = cfg["canonical_pipeline"]["native_k64"]
        a3 = cfg["canonical_pipeline"]["a3f1"]

        assert head["seed"] == 2023
        assert head["batch_size"] == 8
        assert head["epochs"] == 3
        assert head["learning_rate"] == 2.0e-4
        assert head["weight_decay"] == 1.0e-4
        assert head["use_amp"] is False
        assert head["patience"] is None
        assert head["composite_fde_weight"] == 0.25
        assert head["scheduler"] == {"type": "cosine", "eta_min": 2.0e-4}
        assert head["loss_weights"] == {
            "trajectory": 1.0, "endpoint": 1.0, "route_progress_supervision": 1.0
        }
        assert head["structural_fingerprint"] == {"tensors": 257, "params": 2298482}

        assert target["seed"] == 2023
        assert target["batch_size"] == 8
        assert target["epochs"] == 18
        assert target["learning_rate"] == 2.0e-4
        assert target["patience"] == 5
        assert target["composite_fde_weight"] == 0.25
        assert target["beta_assign"] == 1.0
        assert target["scheduler"] == {"type": "cosine", "eta_min": 2.0e-4}
        assert target["loss_weights"] == {
            "trajectory": 1.0,
            "endpoint": 1.5,
            "classification": 0.25,
            "dynamics": 0.05,
            "diversity": 0.05,
        }
        assert target["structural_fingerprint"] == {
            "tensors": 314, "params": 2822517, "fresh_direct_tensors": 57
        }

        assert anchor["seed"] == 2023
        assert anchor["epochs"] == 10
        assert anchor["learning_rate"] == 5.0e-4
        assert anchor["eta_min"] == 5.0e-5
        assert anchor["patience"] == 3
        assert anchor["composite_fde_weight"] == 0.5
        assert anchor["beta_assign"] == 1.0
        assert anchor["structural_output"]["fresh_anchor_tensors"] == 4
        assert anchor["structural_output"]["trainable_params"] == 16770

        assert k64["seed"] == 20260816
        assert k64["batch_size"] == 4
        assert k64["grad_accum"] == 8
        assert k64["warmup_epochs"] == 2
        assert k64["joint_epochs"] == 4
        assert k64["new_query_lr"] == 2.0e-4
        assert k64["shared_decoder_lr"] == 5.0e-5
        assert k64["backbone_lr"] == 1.0e-5

        assert a3["seed"] == 20260816
        assert a3["epoch1_loader_seed"] == 20261825
        assert a3["beta_assign"] == 1.0


def test_fullsize_direct_target_and_anchor_structural_fingerprints() -> None:
    from pathlib import Path
    import yaml
    from wztarf.models import WZTARF, WZTARFConfig

    root = Path(__file__).resolve().parents[1]
    cfg = yaml.safe_load((root / "configs" / "wz.yaml").read_text(encoding="utf-8"))

    head_cfg = dict(cfg["model"])
    head_cfg.update(use_dense_progress_repair=True, use_direct_decoder=False)
    head = WZTARF(WZTARFConfig(**head_cfg))
    assert len(head.state_dict()) == 257
    assert sum(p.numel() for p in head.parameters()) == 2298482

    legacy_cfg = dict(cfg["model"])
    legacy_cfg.update(use_dense_progress_repair=False, use_direct_decoder=False)
    legacy = WZTARF(WZTARFConfig(**legacy_cfg))
    assert len(legacy.state_dict()) == 247
    assert sum(p.numel() for p in legacy.parameters()) == 2243801
    added = sorted(set(head.state_dict()) - set(legacy.state_dict()))
    assert len(added) == 10
    assert sum(head.state_dict()[key].numel() for key in added) == 54681
    assert all(
        key.startswith((
            "route_progress.hard_route_geometry_encoder.",
            "route_progress.dense_progress_fusion.",
            "route_progress.dense_progress_residual_head.",
        ))
        for key in added
    )

    target_cfg = dict(cfg["model"])
    target_cfg.update(
        use_dense_progress_repair=True,
        use_direct_decoder=True,
        use_direct_anchor_calibration=False,
        use_direct_longitudinal_repair=False,
    )
    target = WZTARF(WZTARFConfig(**target_cfg))
    assert len(target.state_dict()) == 314
    assert sum(p.numel() for p in target.parameters()) == 2822517
    assert not any("anchor_correction_head" in k for k in target.state_dict())

    anchor_cfg = dict(cfg["model"])
    anchor_cfg.update(
        use_direct_decoder=True,
        use_direct_anchor_calibration=True,
        use_direct_longitudinal_repair=False,
    )
    anchor = WZTARF(WZTARFConfig(**anchor_cfg))
    assert len(anchor.state_dict()) == 318
    assert sum(p.numel() for p in anchor.parameters()) == 2839287
    anchor_keys = [k for k in anchor.state_dict() if "anchor_correction_head" in k]
    assert len(anchor_keys) == 4
    assert sum(anchor.state_dict()[k].numel() for k in anchor_keys) == 16770


def test_k64_and_a3_force_historical_zero_aux_dropout() -> None:
    import inspect
    from wztarf.pipeline.direct import train_a3f1_one_epoch, train_native_k64_adaptation

    for fn in (train_native_k64_adaptation, train_a3f1_one_epoch):
        src = inspect.getsource(fn)
        assert "aux_dropout_controls=0.0" in src
        assert "aux_dropout_gaze=0.0" in src
        assert "aux_dropout_workers=0.0" in src
    assert "_zero_training_dropout(model)" in inspect.getsource(train_native_k64_adaptation)
