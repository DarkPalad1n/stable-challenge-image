"""Multi-crop data loading for SwAV on the challenge's train.parquet."""
import stable_pretraining as spt
import torch
from stable_pretraining.data import transforms

from training_data import load_training_data

GLOBAL_SIZE, LOCAL_SIZE = 96, 48
N_GLOBAL, N_LOCAL = 2, 4


def _crop(size, scale):
    return transforms.Compose(
        transforms.RGB(),
        transforms.RandomResizedCrop((size, size), scale=scale),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(
            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1, p=0.8
        ),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToImage(**spt.data.static.ImageNet),
    )


def multicrop_transform():
    views = {f"global_{i}": _crop(GLOBAL_SIZE, (0.14, 1.0)) for i in range(N_GLOBAL)}
    views |= {f"local_{i}": _crop(LOCAL_SIZE, (0.05, 0.14)) for i in range(N_LOCAL)}
    return transforms.MultiViewTransform(views)


class ChallengeImages(torch.utils.data.Dataset):
    """The 7,500 training images, each returned as a dict of crops."""

    def __init__(self, transform):
        self.images = load_training_data()
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        row = self.images[index]
        return self.transform({"image": row["image"], "label": row["label"]})


def train_loader(batch_size=128, num_workers=0):
    return torch.utils.data.DataLoader(
        ChallengeImages(multicrop_transform()),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    batch = next(iter(train_loader(batch_size=8)))
    for name, view in batch.items():
        print(name, view["image"].shape)
