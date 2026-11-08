"""Checks for the SwAV multi-crop loader: crop shapes, no labels, and seeded reproducibility."""

import numpy as np
import pytest
import torch
from conftest import images
from datasets import Dataset, Features, Image, Value

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
