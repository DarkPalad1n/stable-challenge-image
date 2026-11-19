"""Shift test: the challenge-style score on degraded copies of the labelled training images.

    uv run shift_score.py runs/*/model.onnx [--data data/train.parquet] [--out shift_scores.json]

Why: ``challenge_score.py`` scores clean training images only. On 2026-11-19 our trained encoders lost
4 to 25 points when the same images were noisy, blurred, downsampled or JPEG-compressed, while an
untrained encoder hardly changed. They seem to rely on fine detail that such shifts destroy. The
public pre-release evaluator (galilai-group/stable-challenge-image-test, ``evaluate.py``) also has
evaluation splits named ``shift_*``, for example noisy images, which it skipped when scoring; whether
the leaderboard uses such splits is not known. This script is a second local signal next to
``challenge_score.py``: how well a model's score holds up when the images get worse.

Every shift is applied to every labelled image with a fixed seed, so all models see the same degraded
images. The probes of ``challenge_score.py`` are then fit and tested on the degraded copies alone, as
if each shift were an evaluation split of its own. "mean" averages the score over the shifts, without
the clean images. The shifts and their strengths are our choice, not the leaderboard's.

Labels are read here only, for evaluation; training never sees them.
"""

import argparse
import io
import json

import numpy as np
import onnxruntime as ort
from datasets import Dataset
from PIL import Image, ImageFilter

import evaluate
from challenge_score import DOMAIN_OF_CLASS, score_features
from training import available_cpus

SHIFT_SEED = 0  # the noise is drawn from this seed, so every model is scored on the same images
SIZE = 96  # pixels; the challenge's image size


def gaussian_noise(sigma):
    """Add Gaussian noise with standard deviation ``sigma`` on the [0, 1] pixel scale."""
    def shift(image, rng):
        pixels = np.asarray(image, dtype=np.float32) / 255 + rng.normal(0, sigma, (SIZE, SIZE, 3))
        return Image.fromarray((np.clip(pixels, 0, 1) * 255).round().astype(np.uint8))
    return shift


def gaussian_blur(radius):
    """Blur with a Gaussian of ``radius`` pixels."""
    return lambda image, rng: image.filter(ImageFilter.GaussianBlur(radius))


def low_resolution(size):
    """Downsample to ``size`` pixels and back up to the full size, which removes fine detail."""
    return lambda image, rng: image.resize((size, size), Image.BILINEAR).resize((SIZE, SIZE), Image.BILINEAR)


def jpeg(quality):
    """Compress as JPEG at ``quality`` (1-95) and decode again."""
    def shift(image, rng):
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=quality)
        return Image.open(io.BytesIO(buffer.getvalue())).convert("RGB")
    return shift


SHIFTS = {
    "noise-0.05": gaussian_noise(0.05),
    "noise-0.10": gaussian_noise(0.10),
    "blur-1": gaussian_blur(1),
    "blur-2": gaussian_blur(2),
    "lowres-48": low_resolution(48),
    "lowres-32": low_resolution(32),
    "jpeg-20": jpeg(20),
}


def degrade(images, shift, seed=SHIFT_SEED):
    """Return a degraded copy of every image.

    Args:
        images: PIL images of size ``SIZE`` x ``SIZE``.
        shift: Name of one entry of ``SHIFTS``.
        seed: Seed of the random draws (the noise); the same seed gives the same images.
    """
    rng = np.random.default_rng(seed)
    return [SHIFTS[shift](image.convert("RGB"), rng) for image in images]


def labelled_images(data_path):
    """The labelled images of ``data_path`` and their labels, in file order."""
    images = Dataset.from_parquet(str(data_path))
    images = images.select(np.flatnonzero(np.asarray(images["label"]) != evaluate.UNLABELLED))
    return list(images["image"]), np.asarray(images["label"])


def embed_images(model_path, images):
    """Embed PIL images with the ONNX model as ``evaluate.py`` does, using only the CPUs we may use."""
    options = ort.SessionOptions()
    options.intra_op_num_threads = available_cpus()
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    return evaluate.embed(session, {"image": images})


def shift_scores(model_path, data_path, domain_of_class=DOMAIN_OF_CLASS):
    """Score one ONNX model on the clean labelled images and on every shift of them.

    Returns:
        Dict with one ``challenge_score``-style result per entry of ``SHIFTS`` plus "clean", and
        "mean", the score averaged over the shifts.
    """
    images, labels = labelled_images(data_path)
    results = {"clean": score_features(embed_images(model_path, images), labels, domain_of_class)}
    for shift in SHIFTS:
        results[shift] = score_features(embed_images(model_path, degrade(images, shift)), labels, domain_of_class)
    results["mean"] = float(np.mean([results[shift]["score"] for shift in SHIFTS]))
    return results


def print_table(results):
    """One row per model: the clean score, the score under every shift, and their mean."""
    header = ["clean", *SHIFTS, "mean"]
    print("model".ljust(48), *(name.rjust(10) for name in header))
    for model_path, result in results.items():
        values = [result[name]["score"] for name in ["clean", *SHIFTS]] + [result["mean"]]
        print(str(model_path)[-48:].ljust(48), *(f"{value:.4f}".rjust(10) for value in values))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="ONNX models to score")
    parser.add_argument("--data", default="data/train.parquet", help="parquet file with image and label columns")
    parser.add_argument("--out", help="also write all results as JSON to this file")
    args = parser.parse_args()
    results = {model_path: shift_scores(model_path, args.data) for model_path in args.models}
    print_table(results)
    if args.out:
        with open(args.out, "w") as file:
            json.dump(results, file, indent=2)
