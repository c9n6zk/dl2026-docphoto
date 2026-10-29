"""predict.py must give, for one image alone, the number the training loop gives for its row.

A model that scores well in the notebook and badly on the leaderboard has
usually been fed different pixels at prediction time. These tests pin the
three places where that happens: the architecture, the standardiser and the
road from the file to the pixels.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from src import predict, train
from src.data.dataset import _resize_one, pixels_of_image
from src.models.mlp import MLP

SIZE = 16


@pytest.fixture
def toy(tmp_path):
    """A small trained-looking checkpoint (random weights) and five noisy photographs."""
    torch.manual_seed(0)
    net = MLP(in_dim=SIZE * SIZE, hidden=[8, 4], dropout=0.3, prior=0.7)
    for p in net.parameters():                   # move away from the constant start so the test can fail
        p.data += 0.1 * torch.randn_like(p)
    cfg = {"data": {"pixel_size": SIZE}, "model": {"hidden": [8, 4], "dropout": 0.3}}
    ckpt = tmp_path / "toy.pt"
    torch.save({"state_dict": net.state_dict(), "config": cfg, "in_dim": SIZE * SIZE,
                "scaler_mean": 0.45, "scaler_scale": 0.21}, ckpt)
    rng = np.random.default_rng(0)
    paths = []
    for k in range(5):
        h, w = 700 + 40 * k, 1000 - 30 * k           # landscape and near-square, so the padding differs
        path = tmp_path / f"img{k}.jpg"
        Image.fromarray(rng.integers(0, 256, (h, w), dtype=np.uint8)).save(path, quality=90)
        paths.append(path)
    return cfg, ckpt, net, paths


def test_one_image_alone_equals_the_training_loop(toy) -> None:
    cfg, ckpt, net, paths = toy
    model = predict.load_model(ckpt, cfg)
    one = np.array([predict.predict_one(model, p, cfg) for p in paths])

    X = np.stack([predict.preprocess(p, cfg) for p in paths])
    batch = train.predict(net, torch.as_tensor((X - 0.45) / 0.21, dtype=torch.float32))
    np.testing.assert_allclose(one, batch, atol=1e-6)
    assert ((one >= 0) & (one <= 1)).all()


def test_preprocess_takes_the_road_of_the_training_pixels(toy, tmp_path) -> None:
    """Training pixels come from the 768 px cache, not straight from the original."""
    cfg, _, _, paths = toy
    cached = tmp_path / "cached.jpg"
    _resize_one(str(paths[0]), cached, predict.CACHE_SIZE)
    np.testing.assert_array_equal(predict.preprocess(paths[0], cfg), pixels_of_image(cached, SIZE))


def test_a_checkpoint_of_another_size_is_refused(toy) -> None:
    cfg, ckpt, _, _ = toy
    with pytest.raises(SystemExit):
        predict.load_model(ckpt, {"data": {"pixel_size": SIZE * 2}})
