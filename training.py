"""The building blocks of a SwAV training run: device and seeds, one epoch, evaluation.

``main.py`` decides what a run does and in which order; this module implements the steps.
"""

import copy
import os
import random

import numpy as np
import onnxruntime as ort
import torch
from datasets import Dataset

import evaluate
from export_onnx import export_onnx


def resolve_device(requested):
    """Return the requested device, or for "auto" the fastest one available."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def seed_everything(seed):
    """Seed every random number generator that training draws from.

    On CPU a run is then exactly repeatable. GPU kernels (CUDA, MPS) may still differ
    in the last digits between runs.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, loss_function, device, freeze_prototypes, queue=None):
    """Run one pass over the training data and return the mean loss.

    Args:
        model: The ``swav.SwAVModel`` being trained.
        loader: Multi-crop loader from ``swav_data.make_train_loader``.
        optimizer: Optimizer over all model parameters.
        loss_function: Maps the model's crop scores (and ``queue_scores``) to the scalar
            SwAV loss.
        device: Device the model lives on.
        freeze_prototypes: If true, the prototype gradients are discarded, so the
            optimizer leaves the prototypes unchanged (AdamW skips parameters without a
            gradient, weight decay included). The SwAV paper does this in the first epoch.
        queue: Optional ``swav.FeatureQueue``. Every batch is pushed into it; once full,
            its scores enter Sinkhorn before the batch is pushed, so it holds earlier
            batches only.
    """
    model.train()
    total_loss = 0.0
    for crops in loader:
        model.normalize_prototypes()
        crop_projections, crop_scores = model([crop.to(device) for crop in crops])
        queue_scores = None
        if queue is not None:
            if queue.is_full:
                queue_scores = queue.scores(model.prototypes)
            queue.push(crop_projections)
        loss = loss_function(crop_scores, queue_scores=queue_scores)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if freeze_prototypes:
            model.prototypes.weight.grad = None
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


def available_cpus():
    """CPUs this process may use: on a cluster node only the job's share, elsewhere all of them.

    ``os.cpu_count()`` counts every CPU of the machine. A Slurm job is limited to its own
    CPUs, and starting a thread for each of the others would only make the threads compete.
    """
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def embed_labelled_images(model_path, data_path):
    """Embed every labelled image of ``data_path`` with the ONNX model, as evaluate.py does.

    ``evaluate.evaluate`` starts one onnxruntime thread per CPU of the machine. This
    function differs from it in the thread count only: it uses ``available_cpus()``.

    Returns:
        Features [num_labelled, dim] and labels [num_labelled].
    """
    images = Dataset.from_parquet(str(data_path))
    images = images.select(np.flatnonzero(np.asarray(images["label"]) != evaluate.UNLABELLED))
    options = ort.SessionOptions()
    options.intra_op_num_threads = available_cpus()
    session = ort.InferenceSession(str(model_path), options, providers=["CPUExecutionProvider"])
    return evaluate.embed(session, images), np.asarray(images["label"])


def probe_score(model_path, data_path):
    """evaluate.py's score: leave-one-out accuracy of its ridge probe on the labelled images."""
    features, labels = embed_labelled_images(model_path, data_path)
    return float((evaluate.Ridge().fit(features, labels).loo_pred == labels).mean())


def export_and_evaluate(model, onnx_path, data_path):
    """Export the backbone in the submission format and return its probe score.

    The export works on a copy, because ``export_onnx`` moves its model to the CPU and
    into eval mode. It also draws a random test input, so the CPU random state is saved
    and restored: an evaluation must not change the rest of the training run.
    """
    with torch.random.fork_rng(devices=[]):
        export_onnx(copy.deepcopy(model.backbone), str(onnx_path))
    return probe_score(onnx_path, data_path)
