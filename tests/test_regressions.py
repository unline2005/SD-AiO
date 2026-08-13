#!/usr/bin/env python3
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(rel):
    return (ROOT / rel).read_text()


def test_classifier_has_full_binary_metrics_and_distributed_gather():
    src = read("src/train_classifier.py")
    assert "def compute_binary_metrics" in src, "classifier eval needs a reusable full binary metric helper"
    assert "accelerator.gather" in src, "classifier eval must gather predictions/labels from all ranks"
    assert "exact_match" in src, "classifier eval must report exact-match accuracy"
    assert "precision" in src and "recall" in src and "f1" in src, "classifier eval must report precision/recall/F1"
    assert "mask = val_labels[:, c] == 1" not in src, "classifier eval must not score positives only"


def test_train_uses_mse_reconstruction_loss_not_l1():
    src = read("src/train.py")
    assert "loss_mse" in src, "training logs should expose weighted MSE reconstruction loss"
    assert "F.mse_loss" in src, "training must use MSE reconstruction loss"
    assert "F.l1_loss" not in src, "training must not use L1 reconstruction loss after MSE switch"
    assert "loss_l1" not in src, "old loss_l1 variable should be removed to avoid misleading logs"


def test_dataset_supports_dod_gt_lq_pairing_and_online_noise():
    src = read("src/utils/dataset.py")
    assert "def is_denoise_task" in src, "dataset must detect denoise tasks (noise_sigma + same LQ/GT dir)"
    assert "def add_noise_tensor" in src, "dataset must synthesise online Gaussian noise for denoise"
    assert "sigma / 127.5" in src, "noise sigma must be scaled from [0,255] to [-1,1] tensor space"
    assert "'online_noise'" in src, "DODPairedDataset must track online_noise per task"
    assert "crc32" in src, "eval denoise noise must be deterministic per sample (crc32-seeded)"
    assert "build_unified_train_dataset" in src, "dataset must export unified train pool builder"
    assert "build_datasets" in src, "dataset must export per-task eval dataset builder"


def test_condition_registry_keeps_canonical_and_alias():
    src = read("src/cond_module.py")
    registry = src[src.index("MODULE_REGISTRY = {"):]
    assert '"deg-aware": DegAwareModule' in registry, "canonical deg-aware key must map to DegAwareModule"
    assert '"deg_aware_sft": DegAwareModule' in registry, "backward-compat alias must also map to DegAwareModule"
    assert '"none": IdentityConditionModule' in registry, "\"none\" key must exist"
    assert '"simple": SimpleModule' in registry, "\"simple\" key must exist"


def test_train_sets_cond_eval_mode_during_eval():
    src = read("src/train.py")
    eval_fn = src[src.index("def evaluate"):src.index("def main")]
    assert "raw_cond_module.eval()" in eval_fn, "evaluation must switch cond_module to eval mode"
    assert "raw_cond_module.train()" in eval_fn, "evaluation must restore cond_module train mode"


def test_infer_loads_partial_cond_module_checkpoint_safely():
    infer_src = read("src/infer.py")
    assert "strict=False" in infer_src, "infer must load trainable-only cond_module checkpoints with strict=False"

def test_train_full_image_eval():
    src = read("src/train.py")
    eval_body = src[src.index("def evaluate"):src.index("def main")]
    assert "raw_model(val_lq," in eval_body, "evaluate must do full-image inference (no crop, no tile)"
    assert "center-crop" not in eval_body, "evaluate must not center-crop during inference"
    assert "tile_inference" not in eval_body, "evaluate must not use tile-based inference"


if __name__ == "__main__":
    tests = [v for k, v in globals().items() if k.startswith("test_")]
    failures = []
    for test in tests:
        try:
            test()
            print(f"PASS {test.__name__}")
        except AssertionError as e:
            print(f"FAIL {test.__name__}: {e}")
            failures.append(test.__name__)
    if failures:
        raise SystemExit(1)
