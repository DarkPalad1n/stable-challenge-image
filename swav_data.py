"""Multi-crop data loading for SwAV training on the challenge's ``train.parquet``.

Every training image is turned into several random crops, following SwAV's multi-crop
strategy (Caron et al., 2020, section 3.2):

* ``NUM_GLOBAL_CROPS`` large crops at ``GLOBAL_CROP_SIZE`` pixels. SwAV computes the
  cluster assignments (the training targets) from these crops only.
* ``NUM_LOCAL_CROPS`` small crops at ``LOCAL_CROP_SIZE`` pixels. They are cheap extra
  views that predict the targets of the global crops.

Every crop is then flipped, colour-jittered, sometimes turned grey and, if enabled,
blurred and degraded (``Degrade``). The defaults follow the stable-pretraining benchmark;
the SwAV paper uses a stronger colour jitter (0.8, 0.8, 0.8, 0.2) and blurs half of the crops.

The loader yields pixels only: the label column is removed as soon as the data is
loaded, so labels cannot reach the training objective (challenge rule).

Smoke check, prints the shape of every crop in one batch::

    uv run swav_data.py [PATH_TO_TRAIN_PARQUET]
"""

import sys

import numpy as np
import stable_pretraining as spt
import torch
from PIL import Image, ImageFilter
from stable_pretraining.data import transforms

from training_data import load_training_data

GLOBAL_CROP_SIZE = 96  # pixels, the full image resolution
LOCAL_CROP_SIZE = 48  # pixels, half resolution
GLOBAL_CROP_AREA = (0.14, 1.0)  # (min, max) fraction of the image area, from the SwAV paper
LOCAL_CROP_AREA = (0.05, 0.14)  # default; set with main.py --local-crop-area
NUM_GLOBAL_CROPS = 2
NUM_LOCAL_CROPS = 4
DEFAULT_COLOR_JITTER = (0.4, 0.4, 0.2, 0.1)  # brightness, contrast, saturation, hue
COLOR_JITTER_PROBABILITY = 0.8
GRAYSCALE_PROBABILITY = 0.2
BLUR_RADIUS = (0.1, 2.0)  # (min, max) in pixels of the crop, as in the SwAV code
# Ranges of the degradations Degrade draws from, on the pixel scale of the crop.
DEGRADE_NOISE_SIGMA = (0.02, 0.10)  # standard deviation on the [0, 1] scale
DEGRADE_BLUR_RADIUS = (0.5, 2.0)  # pixels
DEGRADE_SCALE = (0.33, 0.75)  # side length after downsampling, as a fraction of the crop


def add_noise(image, sigma):
    """Add Gaussian noise with standard deviation ``sigma`` (on the [0, 1] scale) to a PIL image.

    The noise comes from torch's global generator, like every other random augmentation step.
    """
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255)
    pixels = (pixels + sigma * torch.randn(pixels.shape)).clamp(0, 1)
    return Image.fromarray((pixels.numpy() * 255).round().astype(np.uint8))


def blur(image, radius):
    """Gaussian-blur a PIL image with ``radius`` pixels."""
    return image.filter(ImageFilter.GaussianBlur(radius))


def lower_resolution(image, scale):
    """Downsample a PIL image to ``scale`` of its side length and back up, which removes fine detail."""
    width, height = image.size
    small = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(small, Image.BILINEAR).resize((width, height), Image.BILINEAR)


class Degrade:
    """With probability ``p``, degrade the crop by one of: Gaussian noise, blur, lower resolution.

    Why: encoders trained on clean crops lose 4-25 points of challenge score on noisy, blurred,
    downsampled or JPEG-compressed images (``shift_score.py``); they seem to rely on fine detail.
    Degrading some training crops asks the encoder to find the same prototypes without that detail.
    JPEG compression is left out on purpose, so ``shift_score.py`` keeps one shift the encoder never
    saw. The kind and its strength are drawn uniformly (ranges ``DEGRADE_*``) from torch's global
    generator, like every other augmentation, so a seeded run is repeatable.
    """

    def __init__(self, p):
        """Build the step.

        Args:
            p: Chance that a crop is degraded, in [0, 1].
        """
        self.p = p

    def __call__(self, sample):
        if torch.rand(1).item() >= self.p:
            return sample
        kind = int(torch.randint(3, (1,)).item())
        strength = torch.rand(1).item()

        def draw(low_high):
            return low_high[0] + strength * (low_high[1] - low_high[0])

        image = sample["image"]
        if kind == 0:
            image = add_noise(image, draw(DEGRADE_NOISE_SIGMA))
        elif kind == 1:
            image = blur(image, draw(DEGRADE_BLUR_RADIUS))
        else:
            image = lower_resolution(image, draw(DEGRADE_SCALE))
        sample["image"] = image
        return sample


def crop_augmentation(crop_size, crop_area, color_jitter=DEFAULT_COLOR_JITTER, blur_probability=0.0,
                      degrade_probability=0.0):
    """Build the augmentation pipeline that produces one random crop of an image.

    Args:
        crop_size: Side length of the square output crop, in pixels.
        crop_area: (min, max) fraction of the original image area that the crop covers.
        color_jitter: Strength of the colour jitter as (brightness, contrast, saturation, hue).
        blur_probability: Chance that the crop is Gaussian-blurred. At 0 the blur step is
            left out entirely, so it draws no random numbers.
        degrade_probability: Chance that the crop is degraded by ``Degrade``. At 0 the step is
            left out entirely, so it draws no random numbers.

    Returns:
        A transform that maps a sample ``{"image": PIL image}`` to a sample whose
        ``"image"`` entry is a normalised float tensor of shape [3, crop_size, crop_size].
    """
    brightness, contrast, saturation, hue = color_jitter
    steps = [
        transforms.RGB(),
        transforms.RandomResizedCrop((crop_size, crop_size), scale=crop_area),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue,
                               p=COLOR_JITTER_PROBABILITY),
        transforms.RandomGrayscale(p=GRAYSCALE_PROBABILITY),
    ]
    if blur_probability > 0:
        steps.append(transforms.PILGaussianBlur(sigma=BLUR_RADIUS, p=blur_probability))
    if degrade_probability > 0:
        steps.append(Degrade(degrade_probability))
    # The evaluator normalises with the ImageNet mean and std, so training does the same.
    steps.append(transforms.ToImage(**spt.data.static.ImageNet))
    return transforms.Compose(*steps)


def multicrop_augmentations(color_jitter=DEFAULT_COLOR_JITTER, blur_probability=0.0, local_crop_area=LOCAL_CROP_AREA,
                            degrade_probability=0.0):
    """Return one augmentation per crop, global crops first, then local crops.

    Args:
        color_jitter: See ``crop_augmentation``.
        blur_probability: See ``crop_augmentation``.
        local_crop_area: (min, max) fraction of the image area a local crop covers. Only the
            visible part of the image changes; local crops stay ``LOCAL_CROP_SIZE`` pixels.
        degrade_probability: See ``crop_augmentation``; applies to global and local crops alike.
    """
    global_crop = crop_augmentation(GLOBAL_CROP_SIZE, GLOBAL_CROP_AREA, color_jitter, blur_probability,
                                    degrade_probability)
    local_crop = crop_augmentation(LOCAL_CROP_SIZE, tuple(local_crop_area), color_jitter, blur_probability,
                                   degrade_probability)
    return [global_crop] * NUM_GLOBAL_CROPS + [local_crop] * NUM_LOCAL_CROPS


class UnlabelledImages(torch.utils.data.Dataset):
    """The training images without their labels, each returned as a list of random crops."""

    def __init__(self, parquet_path, crop_augmentations):
        """Load the images and drop the label column.

        Args:
            parquet_path: Path to ``train.parquet``. ``None`` uses ``data/train.parquet``
                and downloads it if missing (see ``training_data.py``).
            crop_augmentations: One transform per crop, as built by ``multicrop_augmentations``.
        """
        self.images = load_training_data(parquet_path).remove_columns("label")
        self.crop_augmentations = crop_augmentations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        """Return the crops of image ``index`` as a list of tensors, in augmentation order."""
        image = self.images[index]["image"]
        return [augment({"image": image})["image"] for augment in self.crop_augmentations]


def make_train_loader(parquet_path, batch_size, seed, num_workers=0, color_jitter=DEFAULT_COLOR_JITTER,
                      blur_probability=0.0, local_crop_area=LOCAL_CROP_AREA, degrade_probability=0.0):
    """Build the shuffled multi-crop loader for SwAV training.

    Reproducibility: ``seed`` fixes the order of the images and, with ``num_workers > 0``,
    the random augmentations inside the worker processes. With ``num_workers=0`` the
    augmentations run in the main process and draw from torch's global generator, so the
    training script must also call ``torch.manual_seed(seed)`` before training.

    Args:
        parquet_path: Path to ``train.parquet``, or ``None`` for the default location.
        batch_size: Number of images per batch.
        seed: Seed for the shuffling (and the worker augmentations, see above).
        num_workers: Worker processes for loading; 0 loads in the main process.
        color_jitter: See ``crop_augmentation``.
        blur_probability: See ``crop_augmentation``.
        local_crop_area: See ``multicrop_augmentations``.
        degrade_probability: See ``crop_augmentation``.

    Returns:
        A DataLoader whose batches are lists of ``NUM_GLOBAL_CROPS + NUM_LOCAL_CROPS``
        tensors: global crops of shape [batch_size, 3, 96, 96] first, then local crops of
        shape [batch_size, 3, 48, 48]. The last incomplete batch is dropped, so every batch
        has the same size and Sinkhorn always balances the same number of samples.
    """
    return torch.utils.data.DataLoader(
        UnlabelledImages(parquet_path, multicrop_augmentations(color_jitter, blur_probability, local_crop_area,
                                                               degrade_probability)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        generator=torch.Generator().manual_seed(seed),
    )


if __name__ == "__main__":
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else None
    torch.manual_seed(0)
    crops = next(iter(make_train_loader(parquet_path, batch_size=8, seed=0)))
    for crop_index, crop_batch in enumerate(crops):
        crop_kind = "global" if crop_index < NUM_GLOBAL_CROPS else "local"
        print(f"crop {crop_index} ({crop_kind}): {tuple(crop_batch.shape)}")
