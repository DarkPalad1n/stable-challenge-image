"""Train a ResNet9 encoder with SwAV on train.parquet and export it for the challenge.

    uv run main.py --data data/train.parquet --output runs/swav-s0 [--epochs 10] [--seed 0] ...

Writes into --output, which must not exist yet:

    config.json     every setting of the run plus the library versions
    history.json    per epoch: mean training loss and, on evaluation epochs, the probe score
    model.onnx      the trained backbone in the submission format
    checkpoint.pt   weights of the full SwAV model (backbone, projector and prototypes)

The probe score is evaluate.py's leave-one-out accuracy on the labelled training images,
computed by ``training.probe_score`` with only the CPUs this process may use.
Labels are used there only, for monitoring; the training objective never sees them.
Training needs no internet, no git and no files besides the Python sources and the data.
"""

import argparse
import functools
import json
import platform
import time
from pathlib import Path

import stable_pretraining as spt
import torch

import swav
import swav_data
import training
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

    augmentation = parser.add_argument_group("augmentation")
    augmentation.add_argument("--color-jitter", type=float, nargs=4, default=list(swav_data.DEFAULT_COLOR_JITTER),
                              metavar=("BRIGHTNESS", "CONTRAST", "SATURATION", "HUE"),
                              help="colour jitter strength; the SwAV paper uses 0.8 0.8 0.8 0.2")
    augmentation.add_argument("--blur-probability", type=float, default=0.0,
                              help="chance that a crop is Gaussian-blurred; the SwAV paper uses 0.5")

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


def make_queue(args, device):
    """The feature queue if ``--queue-length`` asks for one, else None."""
    if args.queue_length == 0:
        return None
    return swav.FeatureQueue(args.queue_length, args.batch_size, swav_data.NUM_GLOBAL_CROPS, device)


def make_loss_function(args):
    """The SwAV loss with this run's settings; it takes the crop scores and the queue scores."""
    return functools.partial(
        swav.swav_loss,
        num_global_crops=swav_data.NUM_GLOBAL_CROPS,
        epsilon=args.epsilon,
        num_iterations=args.sinkhorn_iterations,
        temperature=args.temperature,
    )


def is_evaluation_epoch(epoch, args):
    """The last epoch is always evaluated, others every ``--eval-every`` epochs if set."""
    return epoch == args.epochs or (args.eval_every > 0 and epoch % args.eval_every == 0)


def main(argv=None):
    """Train SwAV as set on the command line and write the run's outputs."""
    args = parse_arguments(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    device = training.resolve_device(args.device)
    training.seed_everything(args.seed)
    save_json(args.output / "config.json", run_config(args, device))

    loader = swav_data.make_train_loader(args.data, args.batch_size, args.seed, args.num_workers,
                                         args.color_jitter, args.blur_probability)
    if len(loader) == 0:
        raise ValueError(f"--batch-size {args.batch_size} is larger than the training set")
    model = swav.SwAVModel(args.prototypes).to(device)
    queue = make_queue(args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_function = make_loss_function(args)

    print(f"training on {device}: {args.epochs} epochs of {len(loader)} steps", flush=True)
    history = []
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        mean_loss = training.train_one_epoch(
            model, loader, optimizer, loss_function, device,
            freeze_prototypes=epoch <= args.freeze_prototypes_epochs,
            queue=queue if epoch >= args.queue_start_epoch else None,
        )
        record = {"epoch": epoch, "loss": round(mean_loss, 4), "train_seconds": round(time.time() - epoch_start, 1)}
        if is_evaluation_epoch(epoch, args):
            score = training.export_and_evaluate(model, args.output / "model.onnx", args.data)
            record["score"] = round(score, 4)
        history.append(record)
        save_json(args.output / "history.json", history)
        print(json.dumps(record), flush=True)

    torch.save(model.state_dict(), args.output / "checkpoint.pt")


if __name__ == "__main__":
    main()
