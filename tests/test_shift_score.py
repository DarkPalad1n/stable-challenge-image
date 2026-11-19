"""Checks for the shift test: repeatable degradations of the right strength, and scores that match challenge_score."""

import numpy as np
import pytest
from conftest import MODEL
from PIL import Image

import challenge_score
import shift_score

ONE_DOMAIN = {label: "only" for label in range(3)}  # the eval_data fixture has classes 0, 1 and 2


def textured_image(seed=0):
    """A 96-pixel RGB image with fine detail, so that every shift visibly changes it."""
    pixels = np.random.default_rng(seed).integers(0, 256, (shift_score.SIZE, shift_score.SIZE, 3), dtype=np.uint8)
    return Image.fromarray(pixels)


def mean_change(original, shifted):
    """Mean absolute pixel difference between two images, on the 0-255 scale."""
    return np.abs(np.asarray(original, dtype=np.float32) - np.asarray(shifted, dtype=np.float32)).mean()


@pytest.mark.parametrize("shift", shift_score.SHIFTS)
def test_every_shift_keeps_the_image_format_and_changes_the_image(shift):
    image = textured_image()
    [shifted] = shift_score.degrade([image], shift)
    assert shifted.size == image.size and shifted.mode == "RGB"
    assert mean_change(image, shifted) > 1


@pytest.mark.parametrize("shift", shift_score.SHIFTS)
def test_the_same_seed_gives_the_same_degraded_images(shift):
    images = [textured_image(seed) for seed in range(3)]
    first, again = shift_score.degrade(images, shift), shift_score.degrade(images, shift)
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in zip(first, again))


@pytest.mark.parametrize("weaker, stronger", [("noise-0.05", "noise-0.10"), ("blur-1", "blur-2"),
                                              ("lowres-48", "lowres-32")])
def test_the_stronger_shift_changes_the_image_more(weaker, stronger):
    image = textured_image()
    [weak], [strong] = shift_score.degrade([image], weaker), shift_score.degrade([image], stronger)
    assert mean_change(image, strong) > mean_change(image, weak)


def test_clean_part_is_the_challenge_score(eval_data):
    result = shift_score.shift_scores(MODEL, eval_data, domain_of_class=ONE_DOMAIN)
    assert result["clean"] == challenge_score.challenge_score(MODEL, eval_data, domain_of_class=ONE_DOMAIN)


def test_result_has_every_shift_and_their_mean(eval_data):
    result = shift_score.shift_scores(MODEL, eval_data, domain_of_class=ONE_DOMAIN)
    assert set(result) == {"clean", "mean", *shift_score.SHIFTS}
    scores = [result[shift]["score"] for shift in shift_score.SHIFTS]
    assert result["mean"] == pytest.approx(np.mean(scores))
    assert all(0 <= score <= 1 for score in scores)

