"""Train a ResNet9 encoder with SwAV on train.parquet and export it for the challenge.

    uv run main.py --data data/train.parquet --output runs/swav-s0 [--epochs 10] [--seed 0] ...

Writes into --output, which must not exist yet:

    config.json     every setting of the run plus the library versions
    history.json    per epoch: mean training loss and, on evaluation epochs, the probe score
    model.onnx      the trained backbone in the submission format
    checkpoint.pt   weights of the full SwAV model (backbone, projector and prototypes)

The probe score is evaluate.py's leave-one-out accuracy on the labelled training images.
Labels are used there only, for monitoring; the training objective never sees them.
Training needs no internet, no git and no files besides the Python sources and the data.
"""

import argparse
import copy
import functools
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import stable_pretraining as spt
import torch

import evaluate
import swav
import swav_data
from export_onnx import export_onnx
from training_data import DEFAULT_PATH


def parse_arguments(argv=None):
    """Read the command line; the defaults are the current baseline configuration."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=DEFAULT_PATH, help="path to train.parquet")
    parser.add_argument("--output", type=Path, required=True, help="new directory for this run's outputs")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    parser.add_argument("--num-workers", type=int, default=0, help="data loading processes; 0 = main process")
    parser.add_argument("--eval-every", type=int, default=0,
                        help="also evaluate every N epochs; 0 = only after the last epoch")

    training = parser.add_argument_group("training")
    training.add_argument("--epochs", type=int, default=10)
    training.add_argument("--batch-size", type=int, default=32)
    training.add_argument("--learning-rate", type=float, default=5e-4)
    training.add_argument("--weight-decay", type=float, default=0.05)

    swav_settings = parser.add_argument_group("SwAV")
    swav_settings.add_argument("--prototypes", type=int, default=300)
    swav_settings.add_argument("--epsilon", type=float, default=0.05, help="Sinkhorn entropy regularisation")
    swav_settings.add_argument("--sinkhorn-iterations", type=int, default=3)
    swav_settings.add_argument("--temperature", type=float, default=0.1, help="softmax temperature of the predictions")
    swav_settings.add_argument("--freeze-prototypes-epochs", type=int, default=1,
                               help="number of initial epochs in which the prototypes are not updated")
    swav_settings.add_argument("--queue-length", type=int, default=0,
                               help="projections of recent batches added to Sinkhorn's input; "
                                    "0 = no queue, otherwise a multiple of --batch-size")
    swav_settings.add_argument("--queue-start-epoch", type=int, default=2,
                               help="epoch in which the queue starts filling; it is used once full")
    return parser.parse_args(argv)


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


def run_config(args, device):
    """Everything needed to repeat this run: all settings plus the library versions."""
    settings = {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()}
    return {
        "settings": settings,
        "device": str(device),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "stable_pretraining": spt.__version__,
    }


def save_json(path, content):
    path.write_text(json.dumps(content, indent=2) + "\n")


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


def main(argv=None):
    args = parse_arguments(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    device = resolve_device(args.device)
    seed_everything(args.seed)
    save_json(args.output / "config.json", run_config(args, device))

    loader = swav_data.make_train_loader(args.data, args.batch_size, args.seed, args.num_workers)
    if len(loader) == 0:
        raise ValueError(f"--batch-size {args.batch_size} is larger than the training set")
    model = swav.SwAVModel(args.prototypes).to(device)
    queue = None
    if args.queue_length > 0:
        queue = swav.FeatureQueue(args.queue_length, args.batch_size, swav_data.NUM_GLOBAL_CROPS, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = functools.partial(
        swav.swav_loss,
        num_global_crops=swav_data.NUM_GLOBAL_CROPS,
        epsilon=args.epsilon,
        num_iterations=args.sinkhorn_iterations,
        temperature=args.temperature,
    )

    print(f"training on {device}: {args.epochs} epochs of {len(loader)} steps", flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        freeze_prototypes = epoch <= args.freeze_prototypes_epochs
        epoch_queue = queue if epoch >= args.queue_start_epoch else None
        mean_loss = train_one_epoch(model, loader, optimizer, loss_function, device, freeze_prototypes, epoch_queue)
        record = {"epoch": epoch, "loss": round(mean_loss, 4), "train_seconds": round(time.time() - epoch_start, 1)}
        if epoch == args.epochs or (args.eval_every > 0 and epoch % args.eval_every == 0):
            record["score"] = round(export_and_evaluate(model, args.output / "model.onnx", args.data), 4)
        history.append(record)
        save_json(args.output / "history.json", history)
        print(json.dumps(record), flush=True)

    torch.save(model.state_dict(), args.output / "checkpoint.pt")


if __name__ == "__main__":
    main()
