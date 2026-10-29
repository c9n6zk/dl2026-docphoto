"""Early stopping must choose on training units, and report on units it never chose with.

If the best epoch were picked on the validation rows, the validation score
would be the minimum over epochs of a noisy curve: optimistic by construction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, train
from src.data.splits import build_splits


@pytest.fixture
def toy(tmp_path, monkeypatch):
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(config, "ROOT", tmp_path)          # the checkpoint lands in tmp_path/models
    rng = np.random.default_rng(0)
    n = 120                                                # 6 strict units of 20 rows
    df = pd.DataFrame({
        "image_id": [f"ph_{i:08x}" for i in range(n)],
        "group_strict": [i // 20 for i in range(n)],
        "usability": rng.uniform(0.2, 1.0, n),
        "eval_ok": 1,
    })
    X = rng.normal(size=(n, 16)).astype(np.float32)
    cfg = {
        "name": "toy_es", "seed": 1,
        "split": {"group_column": "group_strict", "val_fraction": 0.2, "seed": 3},
        "model": {"hidden": [8], "dropout": 0.0},
        "train": {"epochs": 6, "batch_size": 16, "lr": 1e-3, "weight_decay": 0.0, "log_every": 100,
                  "early_stopping": {"val_fraction": 0.2, "seed": 0}},
        "wandb": {"project": "test", "tags": []},
    }
    train_idx, val_idx = build_splits(df, "group_strict", 0.2, 3)
    return cfg, X, df, train_idx, val_idx


def test_held_out_units_are_disjoint_from_fit_and_validation(toy) -> None:
    cfg, X, df, train_idx, val_idx = toy
    r = train.train_mlp(cfg, X, df, train_idx, val_idx, device="cpu", log=lambda *a: None)
    fit_u = set(df.iloc[r["fit_index"]]["group_strict"])
    es_u = set(df.iloc[r["es_index"]]["group_strict"])
    val_u = set(df.iloc[val_idx]["group_strict"])
    assert not fit_u & es_u and not fit_u & val_u and not es_u & val_u
    assert sorted(np.concatenate([r["fit_index"], r["es_index"]])) == sorted(train_idx)


def test_the_kept_weights_are_the_epoch_with_the_best_early_stop_score(toy) -> None:
    cfg, X, df, train_idx, val_idx = toy
    r = train.train_mlp(cfg, X, df, train_idx, val_idx, device="cpu", log=lambda *a: None)
    h = r["history"]
    assert r["best_epoch"] == int(h["es_rmse"].idxmin()) + 1
    # the reported validation score is the one of that epoch, whatever the validation curve did
    assert r["val_rmse"] == pytest.approx(h["val_rmse"].iloc[r["best_epoch"] - 1])
    assert np.isclose(train.validate(r["model"], _val_tensor(toy, r), df["usability"].to_numpy()[val_idx],
                                     r["ok"]), r["val_rmse"], atol=1e-6)


def test_without_the_option_nothing_changes(toy) -> None:
    cfg, X, df, train_idx, val_idx = toy
    del cfg["train"]["early_stopping"]
    r = train.train_mlp(cfg, X, df, train_idx, val_idx, device="cpu", log=lambda *a: None)
    assert r["best_epoch"] == cfg["train"]["epochs"]
    assert len(r["fit_index"]) == len(train_idx) and r["es_index"] is None
    assert "es_rmse" not in r["history"].columns


def _val_tensor(toy, r):
    """The validation rows as the training function saw them: standardised on the fit rows."""
    _, X, _, _, val_idx = toy
    fit = X[r["fit_index"]]
    mean, scale = float(fit.mean()), float(fit.std()) + 1e-8
    return train.to_tensor((X[val_idx] - mean) / scale, "cpu")


# --------------------------------------------------------------- the CNN loop of lab 4

@pytest.fixture
def toy_cnn(monkeypatch, tmp_path):
    from src import train_cnn  # noqa: F401  (imported here so a broken import fails the fixture, not the module)

    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.setattr(config, "ROOT", tmp_path)
    rng = np.random.default_rng(0)
    n = 180                                                # 6 strict units of 30 rows
    df = pd.DataFrame({
        "image_id": [f"ph_{i:08x}" for i in range(n)],
        "group_strict": [i // 30 for i in range(n)],
        "usability": rng.uniform(0.2, 1.0, n),
        "eval_ok": 1,
        "family": "toy", "photographer": "P0",
    })
    X = rng.integers(0, 256, (n, 1, 16, 16), dtype=np.uint8)
    cfg = {
        "name": "toy_cnn_es", "seed": 1,
        "data": {"kind": "grey", "pixel_size": 16},
        "split": {"group_column": "group_strict", "val_fraction": 0.2, "seed": 3},
        "model": {"type": "small_cnn", "widths": [4, 8], "stem_stride": 1, "stem_kernel": 3, "pool": "avg",
                  "norm": "batch", "dropout": 0.0},
        "train": {"epochs": 5, "batch_size": 16, "lr": 1e-3, "weight_decay": 0.0, "clip_norm": 1.0,
                  "schedule": "none", "loss": "mse", "augment": "none",
                  "early_stopping": {"val_fraction": 0.2, "seed": 0}},
        "wandb": {"project": "test", "tags": []},
    }
    train_idx, val_idx = build_splits(df, "group_strict", 0.2, 3)
    return cfg, X, df, train_idx, val_idx


def test_cnn_early_stopping_chooses_on_held_out_training_units(toy_cnn) -> None:
    from src import train_cnn

    cfg, X, df, train_idx, val_idx = toy_cnn
    r = train_cnn.train_model(cfg, X, df, train_idx, val_idx, device="cpu", save=False, log=lambda *a: None)
    fit_u = set(df.iloc[r["fit_index"]]["group_strict"])
    es_u = set(df.iloc[r["es_index"]]["group_strict"])
    val_u = set(df.iloc[val_idx]["group_strict"])
    assert not fit_u & es_u and not fit_u & val_u and not es_u & val_u
    h = r["history"]
    assert r["selected_epoch"] == int(h["es_rmse"].idxmin()) + 1      # the validation curve did not decide
    ok = r["ok"]
    y_val = df["usability"].to_numpy()[val_idx]
    reported = float(np.sqrt(np.mean((y_val[ok] - r["val_pred"][ok]) ** 2)))
    assert reported == pytest.approx(h["val_rmse"].iloc[r["selected_epoch"] - 1], abs=1e-6)


def test_cnn_without_early_stopping_is_unchanged(toy_cnn) -> None:
    from src import train_cnn

    cfg, X, df, train_idx, val_idx = toy_cnn
    del cfg["train"]["early_stopping"]
    r = train_cnn.train_model(cfg, X, df, train_idx, val_idx, device="cpu", save=False, log=lambda *a: None)
    assert r["selected_epoch"] == cfg["train"]["epochs"] and r["es_index"] is None
    assert len(r["fit_index"]) == len(train_idx) and "es_rmse" not in r["history"].columns
