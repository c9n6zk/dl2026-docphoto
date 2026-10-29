"""The feature of my own has to see text and ignore grain."""
from __future__ import annotations

import cv2
import numpy as np

from src.features import handcrafted


def page(rows: int = 0, noise: float = 0.0, seed: int = 0) -> np.ndarray:
    """A white 768 x 576 page with `rows` lines of 1 px dark strokes, plus Gaussian grain."""
    img = np.full((768, 576), 235.0)
    for k in range(rows):
        r = 60 + 14 * k
        img[r:r + 8, 60:500:3] = 70.0               # a dashed line of letter-high strokes
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def test_blank_page_has_no_ink() -> None:
    assert handcrafted.ink_fraction(page(rows=0)) == 0.0


def test_grain_alone_is_not_ink() -> None:
    assert handcrafted.ink_fraction(page(rows=0, noise=6.0)) == 0.0


def test_more_text_means_more_ink_and_grain_does_not_change_it_much() -> None:
    few, many = handcrafted.ink_fraction(page(rows=5)), handcrafted.ink_fraction(page(rows=40))
    assert 0 < few < many < 0.5
    assert abs(handcrafted.ink_fraction(page(rows=40, noise=6.0)) - many) < 0.2 * many


def test_it_is_one_of_the_baseline_features() -> None:
    row = {name: fn(page(rows=10)) for name, fn in handcrafted.FEATURES.items()}
    assert "ink_fraction" in row and len(row) == 8
