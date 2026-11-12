"""Checks for the training script as a whole: outputs, reproducibility, queue settings, offline retrain."""

import json
import shutil
import subprocess
import sys

import pytest
from conftest import ROOT

import main

# The offline retrain receives only the Python sources, so main.py must run with exactly these.
TRAINING_SOURCES = ["main.py", "training.py", "swav.py", "swav_data.py", "training_data.py", "export_onnx.py",
                    "evaluate.py"]
FAST_SETTINGS = ["--device", "cpu", "--batch-size", "8", "--prototypes", "10"]


def train(data_path, output_dir, *extra_arguments):
    """Run main.py in-process with fast settings and return the per-epoch history."""
    main.main(["--data", str(data_path), "--output", str(output_dir), *FAST_SETTINGS, *extra_arguments])
    return json.loads((output_dir / "history.json").read_text())


def test_defaults_are_the_configuration_we_hand_in():
    # The offline retrain runs main.py without options, so these values are the submitted recipe.
    settings = vars(main.parse_arguments(["--output", "run"]))
    expected = {
        "seed": 0, "num_workers": 0,
        "epochs": 200, "batch_size": 128, "learning_rate": 5e-4, "weight_decay": 0.05,
        "color_jitter": [0.4, 0.4, 0.2, 0.1], "blur_probability": 0.0,
        "prototypes": 300, "epsilon": 0.05, "sinkhorn_iterations": 3, "temperature": 0.1,
        "freeze_prototypes_epochs": 1, "queue_length": 0,
    }
    assert {name: settings[name] for name in expected} == expected


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


def test_evaluated_models_are_kept_on_request(eval_data, tmp_path):
    run_dir = tmp_path / "run"
    train(eval_data, run_dir, "--epochs", "3", "--eval-every", "2", "--keep-evaluated-models")
    kept = sorted(path.name for path in run_dir.glob("model-epoch*.onnx"))
    assert kept == ["model-epoch0002.onnx", "model-epoch0003.onnx"]
    assert (run_dir / "model-epoch0003.onnx").read_bytes() == (run_dir / "model.onnx").read_bytes()
    assert (run_dir / "model-epoch0002.onnx").read_bytes() != (run_dir / "model.onnx").read_bytes()
    assert json.loads((run_dir / "config.json").read_text())["settings"]["keep_evaluated_models"] is True


def test_kept_model_of_an_epoch_is_the_model_of_a_run_that_ends_there(eval_data, tmp_path):
    # This is why one long run gives the whole training curve.
    train(eval_data, tmp_path / "long", "--epochs", "2", "--eval-every", "1", "--keep-evaluated-models")
    train(eval_data, tmp_path / "short", "--epochs", "1")
    assert (tmp_path / "long" / "model-epoch0001.onnx").read_bytes() == (tmp_path / "short" / "model.onnx").read_bytes()


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


def test_queue_changes_training_from_its_start_epoch_only(eval_data, tmp_path):
    # 41 images at batch size 8 give 5 steps per epoch; a queue of 16 is full after 2 of them.
    without_queue = train(eval_data, tmp_path / "without", "--epochs", "2")
    with_queue = train(eval_data, tmp_path / "with", "--epochs", "2", "--queue-length", "16", "--queue-start-epoch", "2")
    assert with_queue[0]["loss"] == without_queue[0]["loss"]
    assert with_queue[1]["loss"] != without_queue[1]["loss"]


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


def test_augmentation_settings_reach_the_loader(eval_data, tmp_path):
    default = train(eval_data, tmp_path / "default", "--epochs", "1")
    blurred = train(eval_data, tmp_path / "blurred", "--epochs", "1", "--blur-probability", "1")
    jittered = train(eval_data, tmp_path / "jittered", "--epochs", "1", "--color-jitter", "0.8", "0.8", "0.8", "0.2")
    assert blurred[0]["loss"] != default[0]["loss"]
    assert jittered[0]["loss"] != default[0]["loss"]
