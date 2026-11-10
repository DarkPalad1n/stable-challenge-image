"""Checks for the SwAV multi-crop loader: crop shapes, no labels, seeded reproducibility, augmentations."""

import numpy as np
import pytest
import torch
from conftest import images
from datasets import Dataset, Features, Image, Value
from PIL import Image as PILImage
from stable_pretraining.data import transforms

import swav_data

NUM_IMAGES = 10
BATCH_SIZE = 4


@pytest.fixture
def labelled_parquet(tmp_path):
    """A small train.parquet look-alike in which every image carries a label."""
    path = tmp_path / "train.parquet"
    labels = list(range(NUM_IMAGES))
    Dataset.from_dict(
        {"image": images(labels, np.random.default_rng(0)), "label": labels},
        features=Features({"image": Image(), "label": Value("int64")}),
    ).to_parquet(path)
    return str(path)


def first_batch(parquet_path, seed):
    """Seed torch the way the training script does, then return the first batch."""
    torch.manual_seed(seed)
    loader = swav_data.make_train_loader(parquet_path, batch_size=BATCH_SIZE, seed=seed)
    return next(iter(loader))


def test_batch_holds_global_crops_then_local_crops(labelled_parquet):
    crops = first_batch(labelled_parquet, seed=0)
    global_shape = (BATCH_SIZE, 3, swav_data.GLOBAL_CROP_SIZE, swav_data.GLOBAL_CROP_SIZE)
    local_shape = (BATCH_SIZE, 3, swav_data.LOCAL_CROP_SIZE, swav_data.LOCAL_CROP_SIZE)
    expected_shapes = [global_shape] * swav_data.NUM_GLOBAL_CROPS + [local_shape] * swav_data.NUM_LOCAL_CROPS
    assert [tuple(crop_batch.shape) for crop_batch in crops] == expected_shapes
    assert all(crop_batch.dtype == torch.float32 for crop_batch in crops)


def test_labels_never_reach_the_loader(labelled_parquet):
    dataset = swav_data.UnlabelledImages(labelled_parquet, swav_data.multicrop_augmentations())
    assert dataset.images.column_names == ["image"]
    assert all(isinstance(crop, torch.Tensor) for crop in dataset[0])


def test_same_seed_gives_identical_batches(labelled_parquet):
    batch_a, batch_b = first_batch(labelled_parquet, seed=0), first_batch(labelled_parquet, seed=0)
    assert all(torch.equal(crop_a, crop_b) for crop_a, crop_b in zip(batch_a, batch_b))


def test_different_seeds_give_different_batches(labelled_parquet):
    batch_a, batch_b = first_batch(labelled_parquet, seed=0), first_batch(labelled_parquet, seed=1)
    assert not torch.equal(batch_a[0], batch_b[0])


def test_incomplete_last_batch_is_dropped(labelled_parquet):
    torch.manual_seed(0)
    loader = swav_data.make_train_loader(labelled_parquet, batch_size=BATCH_SIZE, seed=0)
    assert len(loader) == NUM_IMAGES // BATCH_SIZE
    assert all(len(crops[0]) == BATCH_SIZE for crops in loader)


def augmentation_steps(**settings):
    """The types of the steps in a global-crop pipeline built with ``settings``."""
    pipeline = swav_data.crop_augmentation(swav_data.GLOBAL_CROP_SIZE, swav_data.GLOBAL_CROP_AREA, **settings)
    return pipeline.args


def test_blur_is_left_out_unless_enabled():
    def blur_steps(**settings):
        return [step for step in augmentation_steps(**settings) if isinstance(step, transforms.PILGaussianBlur)]

    assert blur_steps() == []
    assert blur_steps(blur_probability=0.0) == []
    [blur] = blur_steps(blur_probability=0.5)
    assert blur.p == 0.5 and tuple(blur.sigma) == swav_data.BLUR_RADIUS


def test_color_jitter_strength_is_configurable():
    [jitter] = [step for step in augmentation_steps(color_jitter=(0.8, 0.8, 0.8, 0.2))
                if isinstance(step, transforms.ColorJitter)]
    ranges = [jitter.brightness, jitter.contrast, jitter.saturation, jitter.hue]
    assert ranges == [pytest.approx((0.2, 1.8))] * 3 + [pytest.approx((-0.2, 0.2))]


def neighbour_difference(crop):
    """Mean absolute difference between horizontally adjacent pixels: high for sharp crops."""
    return (crop[..., 1:] - crop[..., :-1]).abs().mean().item()


def test_blur_smooths_the_crop_and_changes_nothing_else():
    # Blur is the last random step, so with the same seed both pipelines produce the same
    # crop up to the blur: any difference comes from the blur alone.
    noise = PILImage.fromarray(np.random.default_rng(0).integers(0, 256, (96, 96, 3), dtype=np.uint8))
    for seed in range(5):
        crops = []
        for blur_probability in (0.0, 1.0):
            torch.manual_seed(seed)
            augment = swav_data.crop_augmentation(swav_data.GLOBAL_CROP_SIZE, swav_data.GLOBAL_CROP_AREA,
                                                  blur_probability=blur_probability)
            crops.append(augment({"image": noise})["image"])
        sharp, blurred = crops
        assert neighbour_difference(blurred) < neighbour_difference(sharp)
        torch.testing.assert_close(blurred.mean(), sharp.mean(), atol=0.05, rtol=0)
