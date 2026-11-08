"""Checks for the training script: outputs, reproducibility, prototype freezing, queue, offline retrain."""

import json
import shutil
import subprocess
import sys

import pytest
import torch
from conftest import ROOT

import main
import swav
import swav_data

# The offline retrain receives only the Python sources, so main.py must run with exactly these.
TRAINING_SOURCES = ["main.py", "swav.py", "swav_data.py", "training_data.py", "export_onnx.py", "evaluate.py"]
FAST_SETTINGS = ["--device", "cpu", "--batch-size", "8", "--prototypes", "10"]


def train(data_path, output_dir, *extra_arguments):
    """Run main.py in-process with fast settings and return the per-epoch history."""
    main.main(["--data", str(data_path), "--output", str(output_dir), *FAST_SETTINGS, *extra_arguments])
    return json.loads((output_dir / "history.json").read_text())


def test_run_writes_every_output(eval_data, tmp_path):
    run_dir = tmp_path / "run"
    history = train(eval_data, run_dir, "--epochs", "2", "--eval-every", "1")
    assert {path.name for path in run_dir.iterdir()} == {"config.json", "history.json", "model.onnx", "checkpoint.pt"}
    assert [record["epoch"] for record in history] == [1, 2]
    assert all(0 <= record["score"] <= 1 for record in history)
    config = json.loads((run_dir / "config.json").read_text())
    assert config["settings"]["seed"] == 0 and config["device"] == "cpu"


def test_last_epoch_is_always_evaluated(eval_data, tmp_path):
    history = train(eval_data, tmp_path / "run", "--epochs", "2")
    assert "score" not in history[0] and "score" in history[1]


def test_existing_run_is_never_overwritten(eval_data, tmp_path):
    (tmp_path / "run").mkdir()
    with pytest.raises(FileExistsError):
        train(eval_data, tmp_path / "run", "--epochs", "1")


def test_same_seed_reproduces_the_run(eval_data, tmp_path):
    first = train(eval_data, tmp_path / "first", "--epochs", "2")
    second = train(eval_data, tmp_path / "second", "--epochs", "2")
    assert [record["loss"] for record in first] == [record["loss"] for record in second]
    assert first[-1]["score"] == second[-1]["score"]


def test_evaluating_does_not_change_training(eval_data, tmp_path):
    evaluated_each_epoch = train(eval_data, tmp_path / "each", "--epochs", "2", "--eval-every", "1")
    evaluated_at_end = train(eval_data, tmp_path / "end", "--epochs", "2")
    assert [record["loss"] for record in evaluated_each_epoch] == [record["loss"] for record in evaluated_at_end]


@pytest.mark.parametrize("freeze_prototypes", [True, False])
def test_prototypes_move_only_when_not_frozen(eval_data, freeze_prototypes):
    torch.manual_seed(0)
    model = swav.SwAVModel(num_prototypes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.05)
    loader = swav_data.make_train_loader(str(eval_data), batch_size=8, seed=0)
    prototypes_before = model.prototypes.weight.detach().clone()

    def loss_function(crop_scores, queue_scores):
        return swav.swav_loss(crop_scores, swav_data.NUM_GLOBAL_CROPS, 0.05, 3, 0.1, queue_scores)

    main.train_one_epoch(model, loader, optimizer, loss_function, torch.device("cpu"), freeze_prototypes)
    largest_change = (model.prototypes.weight - prototypes_before).abs().max().item()
    # Re-normalising unit vectors every step moves them by float32 rounding (~1e-8) only;
    # an optimizer step at this learning rate moves them by orders of magnitude more.
    if freeze_prototypes:
        assert largest_change < 1e-6
    else:
        assert largest_change > 1e-3


def test_queue_changes_training_from_its_start_epoch_only(eval_data, tmp_path):
    # 41 images at batch size 8 give 5 steps per epoch; a queue of 16 is full after 2 of them.
    without_queue = train(eval_data, tmp_path / "without", "--epochs", "2")
    with_queue = train(eval_data, tmp_path / "with", "--epochs", "2", "--queue-length", "16", "--queue-start-epoch", "2")
    assert with_queue[0]["loss"] == without_queue[0]["loss"]
    assert with_queue[1]["loss"] != without_queue[1]["loss"]


def test_queue_holds_only_earlier_batches_newest_first(eval_data):
    # With frozen prototypes, a stored projection scores exactly as it did in its own step.
    torch.manual_seed(0)
    model = swav.SwAVModel(num_prototypes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.05)
    loader = swav_data.make_train_loader(str(eval_data), batch_size=8, seed=0)
    queue = swav.FeatureQueue(16, 8, swav_data.NUM_GLOBAL_CROPS, device="cpu")
    steps = []

    def recording_loss(crop_scores, queue_scores):
        steps.append(([scores.detach() for scores in crop_scores], queue_scores))
        return swav.swav_loss(crop_scores, swav_data.NUM_GLOBAL_CROPS, 0.05, 3, 0.1, queue_scores)

    main.train_one_epoch(model, loader, optimizer, recording_loss, torch.device("cpu"), True, queue)
    assert [queue_scores is None for _, queue_scores in steps] == [True, True, False, False, False]
    for step in range(2, len(steps)):
        queue_scores = steps[step][1]
        for crop_index in range(swav_data.NUM_GLOBAL_CROPS):
            expected = torch.cat([steps[step - 1][0][crop_index], steps[step - 2][0][crop_index]])
            torch.testing.assert_close(queue_scores[crop_index], expected)


def test_queue_length_is_checked_before_training(eval_data, tmp_path):
    with pytest.raises(ValueError, match="multiple of the batch size"):
        train(eval_data, tmp_path / "run", "--epochs", "1", "--queue-length", "12")


def test_retrain_works_with_only_the_python_sources(eval_data, tmp_path):
    """Simulates the offline retrain: no git, no pyproject.toml, no data/ folder, a foreign working directory."""
    bare_dir = tmp_path / "bare"
    bare_dir.mkdir()
    for source in TRAINING_SOURCES:
        shutil.copy(ROOT / source, bare_dir / source)
    command = [sys.executable, str(bare_dir / "main.py"), "--data", str(eval_data), "--output", "run",
               "--epochs", "1", *FAST_SETTINGS]
    result = subprocess.run(command, cwd=bare_dir, capture_output=True, text=True, timeout=600)
    assert result.returncode == 0, result.stderr
    assert (bare_dir / "run" / "model.onnx").is_file()
