"""The building blocks of a SwAV training run: device and seeds, one epoch, evaluation.

``main.py`` decides what a run does and in which order; this module implements the steps.
"""

import copy
import random

import numpy as np
import torch

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


def export_and_evaluate(model, onnx_path, data_path):
    """Export the backbone in the submission format and return evaluate.py's probe score.

    The export works on a copy, because ``export_onnx`` moves its model to the CPU and
    into eval mode. It also draws a random test input, so the CPU random state is saved
    and restored: an evaluation must not change the rest of the training run.
    """
    with torch.random.fork_rng(devices=[]):
        export_onnx(copy.deepcopy(model.backbone), str(onnx_path))
    return evaluate.evaluate(str(onnx_path), str(data_path))["score"]
