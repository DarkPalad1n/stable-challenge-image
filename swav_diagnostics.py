"""Diagnostics of trained SwAV runs: how soft the Sinkhorn targets are and how the prototypes are used.

    uv run swav_diagnostics.py runs/*/ [--data data/train.parquet] [--batch-size 32] [--out diagnostics.json]

For the research question why the batch size matters and why Galaxy10 is the weakest domain. Each run
directory must hold the ``config.json`` and ``checkpoint.pt`` that ``main.py`` writes. Per run:

    target entropy   Mean entropy (in nats) of the Sinkhorn targets, computed as in training: on the two
                     global crops, with the run's augmentation, epsilon and iterations, one batch at a
                     time. The maximum, log(prototypes), means a target spread over all prototypes, 0 a
                     one-hot target. ``--batch-size`` replaces the run's batch size, which separates what
                     a model has learned from what the batch size does to Sinkhorn.
    prototypes used  How many prototypes are the best-scoring one for at least one training image
                     (plain images, no augmentation), and the effective number: exp of the entropy of
                     the usage counts, which is lower when a few prototypes take most images.
    per domain       The same for the labelled images of each domain, and how many of a domain's
                     prototypes no other domain uses.

The model runs in evaluation mode, so BatchNorm uses its running statistics; training normalises with
the statistics of each batch, so the target entropy during training can differ slightly. No queue is
used. Labels are read here only, to split the last table by domain; training never sees them.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

import evaluate
import swav
import swav_data
import training
from challenge_score import DOMAIN_OF_CLASS
from training_data import UNLABELLED, load_training_data

SEED = 0  # of the image order and the augmentations, so every run is measured on the same crops
EMBED_BATCH_SIZE = 256


def load_run(run_dir, device):
    """Rebuild the model of a finished run.

    Returns:
        The ``swav.SwAVModel`` with the run's trained weights, in evaluation mode, and the run's settings.
    """
    run_dir = Path(run_dir)
    settings = json.loads((run_dir / "config.json").read_text())["settings"]
    model = swav.SwAVModel(settings["prototypes"])
    model.load_state_dict(torch.load(run_dir / "checkpoint.pt", map_location="cpu"))
    return model.to(device).eval(), settings


@torch.no_grad()
def target_entropy(model, loader, epsilon, num_iterations, device):
    """Mean entropy in nats of the Sinkhorn targets over one pass of ``loader``.

    Args:
        model: ``swav.SwAVModel`` in evaluation mode.
        loader: Multi-crop loader from ``swav_data.make_train_loader``; only its global crops are used,
            because only they get targets.
        epsilon: Sinkhorn entropy regularisation.
        num_iterations: Sinkhorn normalisation rounds.
        device: Device the model lives on.
    """
    entropies = []
    for crops in loader:
        _, crop_scores = model([crop.to(device) for crop in crops[:swav_data.NUM_GLOBAL_CROPS]])
        for scores in crop_scores:
            targets = swav.sinkhorn(scores, epsilon, num_iterations)
            entropies.append(torch.special.entr(targets).sum(dim=1).cpu())
    return torch.cat(entropies).mean().item()


def normalised_images(dataset):
    """The dataset's images as the evaluator feeds them to a model: float32 [N, 3, 96, 96], normalised."""
    pixels = np.stack([np.asarray(image, dtype=np.float32) for image in dataset["image"]]) / 255.0
    return torch.from_numpy(((pixels - evaluate.MEAN) / evaluate.STD).astype(np.float32).transpose(0, 3, 1, 2))


@torch.no_grad()
def best_prototypes(model, images, device):
    """Index of the best-scoring prototype of every image, as an array [num_images]."""
    best = []
    for batch in images.split(EMBED_BATCH_SIZE):
        _, crop_scores = model([batch.to(device)])
        best.append(crop_scores[0].argmax(dim=1).cpu())
    return torch.cat(best).numpy()


def prototype_usage(assignments, num_prototypes):
    """How many prototypes a set of images uses.

    Args:
        assignments: Array [num_images] with the prototype index of every image.
        num_prototypes: Number of prototypes of the model.

    Returns:
        Dict with "images", "prototypes_used" (prototypes with at least one image) and
        "effective_prototypes" (exp of the entropy of the usage counts).
    """
    counts = np.bincount(assignments, minlength=num_prototypes)
    shares = counts[counts > 0] / counts.sum()
    return {"images": int(counts.sum()), "prototypes_used": int((counts > 0).sum()),
            "effective_prototypes": float(np.exp(-(shares * np.log(shares)).sum()))}


def domain_usage(assignments, domains, num_prototypes):
    """Prototype usage of every domain, and how many of its prototypes are its own.

    Args:
        assignments: Array [num_images] with the prototype index of every labelled image.
        domains: Array [num_images] with the domain name of every labelled image.
        num_prototypes: Number of prototypes of the model.

    Returns:
        Dict from domain name to its ``prototype_usage`` plus "exclusive_prototypes": the prototypes
        that this domain uses and no other domain does.
    """
    used_by = {domain: set(assignments[domains == domain].tolist()) for domain in sorted(set(domains))}
    usage = {}
    for domain, used in used_by.items():
        used_by_others = set().union(*(other_used for other, other_used in used_by.items() if other != domain))
        usage[domain] = prototype_usage(assignments[domains == domain], num_prototypes)
        usage[domain]["exclusive_prototypes"] = len(used - used_by_others)
    return usage


def diagnose(run_dir, data_path, batch_size=None, device="cpu", domain_of_class=DOMAIN_OF_CLASS):
    """Compute all diagnostics of one run.

    Args:
        run_dir: Run directory written by ``main.py``.
        data_path: Path to ``train.parquet``.
        batch_size: Batch size for the Sinkhorn targets; None uses the run's own.
        device: Device to compute on.
        domain_of_class: Dict from class id to domain name.

    Returns:
        Dict with "batch_size", "target_entropy", "max_entropy", "prototypes", the ``prototype_usage``
        of all training images, and "per_domain" with the ``domain_usage`` of the labelled images.
    """
    model, settings = load_run(run_dir, device)
    batch_size = batch_size or settings["batch_size"]
    num_prototypes = settings["prototypes"]

    torch.manual_seed(SEED)  # the augmentations draw from torch's global generator
    # Runs from before the augmentation became configurable have no such settings; they used these values.
    loader = swav_data.make_train_loader(
        str(data_path), batch_size, SEED,
        color_jitter=settings.get("color_jitter", swav_data.DEFAULT_COLOR_JITTER),
        blur_probability=settings.get("blur_probability", 0.0))
    result = {"batch_size": batch_size, "prototypes": num_prototypes, "max_entropy": math.log(num_prototypes),
              "target_entropy": target_entropy(model, loader, settings["epsilon"], settings["sinkhorn_iterations"],
                                               device)}

    dataset = load_training_data(str(data_path))
    assignments = best_prototypes(model, normalised_images(dataset), device)
    result.update(prototype_usage(assignments, num_prototypes))
    labels = np.asarray(dataset["label"])
    labelled = labels != UNLABELLED
    domains = np.array([domain_of_class[label] for label in labels[labelled]])
    result["per_domain"] = domain_usage(assignments[labelled], domains, num_prototypes)
    return result


def print_table(results):
    """One row per run: target entropy, overall prototype usage, then used/exclusive prototypes per domain."""
    domains = sorted(next(iter(results.values()))["per_domain"])
    header = ["batch", "entropy", "max", "used", "effective"] + [f"{domain}:used/own" for domain in domains]
    print("run".ljust(40), *(name.rjust(max(len(name), 7)) for name in header))
    for run_dir, result in results.items():
        values = [str(result["batch_size"]), f"{result['target_entropy']:.3f}", f"{result['max_entropy']:.3f}",
                  str(result["prototypes_used"]), f"{result['effective_prototypes']:.1f}"]
        values += [f"{result['per_domain'][domain]['prototypes_used']}/"
                   f"{result['per_domain'][domain]['exclusive_prototypes']}" for domain in domains]
        print(str(run_dir).rstrip("/")[-40:].ljust(40),
              *(value.rjust(max(len(name), 7)) for value, name in zip(values, header)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="run directories written by main.py")
    parser.add_argument("--data", default="data/train.parquet", help="path to train.parquet")
    parser.add_argument("--batch-size", type=int, help="batch size for the Sinkhorn targets; default: each run's own")
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    parser.add_argument("--out", help="also write all results as JSON to this file")
    args = parser.parse_args()
    torch.set_num_threads(training.available_cpus())
    device = training.resolve_device(args.device)
    results = {run_dir: diagnose(run_dir, args.data, args.batch_size, device) for run_dir in args.runs}
    print_table(results)
    if args.out:
        with open(args.out, "w") as file:
            json.dump(results, file, indent=2)
