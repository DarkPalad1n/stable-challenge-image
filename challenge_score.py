"""Local stand-in for the leaderboard score, computed on the labelled training images.

    uv run challenge_score.py runs/*/model.onnx [--data data/train.parquet] [--out scores.json]

The leaderboard's evaluator is hidden. The pre-release repo galilai-group/stable-challenge-image-test
(``evaluate.py``, 2026-10-07) scored a model as the mean of two parts, each averaged over the three
domains (Imagenette, Galaxy10, EuroSAT). This script mirrors both parts on the labelled images of
``train.parquet``:

    in-domain  Leaderboard: per domain, a ridge probe fit on probe images, accuracy on test images.
               Here: per domain, the leave-one-out accuracy of the same ridge probe.
    few-shot   Leaderboard: on classes unseen in training, a ridge probe fit on 10 images per class,
               accuracy on the remaining images, averaged over 5 random episodes.
               Here: the same per domain, on the training classes, over ``EPISODES`` episodes.

Differences from the leaderboard: every class here was seen (unlabelled) during training, so the
few-shot part measures how well 10 examples separate known classes, not transfer to new ones; and all
images are training images. Use the score to compare runs, not to predict leaderboard numbers. Every
run is scored on the same episodes (fixed seed), so differences between runs are paired.

Labels are read here only, for evaluation; training never sees them.
"""

import argparse
import json
import os

import numpy as np
import onnxruntime as ort
from datasets import Dataset

from evaluate import UNLABELLED, Ridge, embed

IMAGENETTE, GALAXY10, EUROSAT = "imagenette", "galaxy10", "eurosat"
# train.parquet does not say which domain a class comes from. This map was read off the labelled images
# of every class and checked with two image statistics: galaxies have a black sky, EuroSAT tiles are
# smooth because they were upscaled from 64 pixels. Class 8 has no labelled images; it is the tenth
# Galaxy10 class by elimination.
DOMAIN_OF_CLASS = {
    **dict.fromkeys([2, 4, 6, 10, 11, 16, 21, 23, 26, 28], IMAGENETTE),
    **dict.fromkeys([0, 3, 8, 12, 13, 18, 19, 20, 25, 29], GALAXY10),
    **dict.fromkeys([1, 5, 7, 9, 14, 15, 17, 22, 24, 27], EUROSAT),
}
SHOTS = 10  # labelled images per class in a few-shot episode, as on the leaderboard
EPISODES = 20  # more than the leaderboard's 5, to make the local score less noisy
EPISODE_SEED = 0


def in_domain_accuracy(features, labels, domains):
    """Leave-one-out accuracy of a ridge probe fit separately within each domain.

    Args:
        features: Array [num_images, dim] of embeddings.
        labels: Array [num_images] of class ids.
        domains: Array [num_images] of domain names.

    Returns:
        Dict from domain name to accuracy.
    """
    accuracies = {}
    for domain in sorted(set(domains)):
        in_domain = domains == domain
        probe = Ridge().fit(features[in_domain], labels[in_domain])
        accuracies[domain] = float((probe.loo_pred == labels[in_domain]).mean())
    return accuracies


def few_shot_split(labels, shots, rng):
    """Draw one episode: ``shots`` support images per class, all other images as queries.

    Classes with at most ``shots`` images are left out, since they would have no query image.

    Returns:
        Two index arrays into ``labels``: support and query.
    """
    support, query = [], []
    for label in np.unique(labels):
        images = rng.permutation(np.flatnonzero(labels == label))
        if len(images) > shots:
            support.append(images[:shots])
            query.append(images[shots:])
    return np.concatenate(support), np.concatenate(query)


def few_shot_accuracy(features, labels, domains, shots=SHOTS, episodes=EPISODES, seed=EPISODE_SEED):
    """Mean accuracy of ridge probes fit on ``shots`` images per class, separately within each domain.

    Args:
        features: Array [num_images, dim] of embeddings.
        labels: Array [num_images] of class ids.
        domains: Array [num_images] of domain names.
        shots: Support images per class.
        episodes: Random support/query splits averaged per domain.
        seed: Seed of the splits; the same seed gives every model the same episodes.

    Returns:
        Dict from domain name to accuracy.
    """
    rng = np.random.default_rng(seed)
    accuracies = {}
    for domain in sorted(set(domains)):
        domain_features, domain_labels = features[domains == domain], labels[domains == domain]
        episode_accuracies = []
        for _ in range(episodes):
            support, query = few_shot_split(domain_labels, shots, rng)
            probe = Ridge().fit(domain_features[support], domain_labels[support])
            episode_accuracies.append((probe.predict(domain_features[query]) == domain_labels[query]).mean())
        accuracies[domain] = float(np.mean(episode_accuracies))
    return accuracies


def available_cpus():
    """CPUs this process may use: on a cluster node only the job's share, elsewhere all of them.

    ``os.cpu_count()`` counts every CPU of the machine. A Slurm job is limited to its own CPUs, and
    starting a thread for each of the others would only make the threads compete.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def embed_labelled_images(model_path, data_path):
    """Embed every labelled image of ``data_path`` with the ONNX model, as evaluate.py does.

    Returns:
        Features [num_labelled, dim] and labels [num_labelled].
    """
    images = Dataset.from_parquet(str(data_path))
    images = images.select(np.flatnonzero(np.asarray(images["label"]) != UNLABELLED))
    options = ort.SessionOptions()
    options.intra_op_num_threads = available_cpus()
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    return embed(session, images), np.asarray(images["label"])


def challenge_score(model_path, data_path, domain_of_class=DOMAIN_OF_CLASS):
    """Score one ONNX model.

    Returns:
        Dict with "score" (mean of the two parts), "in_domain" and "few_shot" (each averaged over the
        domains), and "per_domain" with both parts for every domain.
    """
    features, labels = embed_labelled_images(model_path, data_path)
    domains = np.array([domain_of_class[label] for label in labels])
    in_domain = in_domain_accuracy(features, labels, domains)
    few_shot = few_shot_accuracy(features, labels, domains)
    result = {"in_domain": float(np.mean(list(in_domain.values()))),
              "few_shot": float(np.mean(list(few_shot.values())))}
    result["score"] = (result["in_domain"] + result["few_shot"]) / 2
    result["per_domain"] = {domain: {"in_domain": in_domain[domain], "few_shot": few_shot[domain]}
                            for domain in in_domain}
    return result


def print_table(results):
    """One row per model: overall score, both parts, and both parts for every domain."""
    domains = sorted(next(iter(results.values()))["per_domain"])
    header = ["score", "in_domain", "few_shot"] + [f"{part}:{domain}" for domain in domains
                                                     for part in ("id", "fs")]
    print("model".ljust(48), *(name.rjust(len(name)) for name in header))
    for model_path, result in results.items():
        values = [result["score"], result["in_domain"], result["few_shot"]]
        values += [result["per_domain"][domain][part] for domain in domains for part in ("in_domain", "few_shot")]
        print(str(model_path)[-48:].ljust(48), *(f"{value:.4f}".rjust(len(name)) for value, name in zip(values, header)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("models", nargs="+", help="ONNX models to score")
    parser.add_argument("--data", default="data/train.parquet", help="parquet file with image and label columns")
    parser.add_argument("--out", help="also write all results as JSON to this file")
    args = parser.parse_args()
    results = {model_path: challenge_score(model_path, args.data) for model_path in args.models}
    print_table(results)
    if args.out:
        with open(args.out, "w") as file:
            json.dump(results, file, indent=2)
