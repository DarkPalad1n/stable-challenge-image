"""Checks for the summary of the cluster runs, on run directories written by hand."""

import json

import pytest

from cluster import summarize


def write_run(runs_dir, name, seed, scores_by_epoch, epochs):
    """Write the two files summarize.py reads from a run: its history and its number of epochs."""
    run_dir = runs_dir / f"hydra-{name}-s{seed}"
    run_dir.mkdir(parents=True)
    history = [{"epoch": epoch, "loss": 1.0} for epoch in range(1, max(scores_by_epoch) + 1)]
    for epoch, score in scores_by_epoch.items():
        history[epoch - 1]["score"] = score
    (run_dir / "history.json").write_text(json.dumps(history))
    (run_dir / "config.json").write_text(json.dumps({"settings": {"epochs": epochs}}))


def write_scores(path, scores_by_model):
    path.write_text(json.dumps({model: {"score": score} for model, score in scores_by_model.items()}))


def test_only_the_last_epoch_of_finished_runs_is_summarised(tmp_path):
    write_run(tmp_path, "b128-e200", 0, {5: 0.9, 10: 0.5}, epochs=10)  # the best epoch (0.9) must not count
    write_run(tmp_path, "b128-e200", 1, {5: 0.6, 10: 0.7}, epochs=10)
    write_run(tmp_path, "b128-e200", 2, {5: 0.8}, epochs=10)  # still running
    (tmp_path / "swav-other-run").mkdir()
    (summary,) = summarize.summarize(tmp_path)
    assert summary["name"] == "b128-e200" and summary["seeds"] == [0, 1]
    assert summary["main_score"] == pytest.approx({"mean": 0.6, "std": 0.02 ** 0.5, "min": 0.5, "max": 0.7})
    assert summary["challenge_score"] is None


def test_configuration_names_may_contain_dashes_and_seed_like_parts(tmp_path):
    write_run(tmp_path, "b128-lr1e-3-s2x", 4, {1: 0.5}, epochs=1)
    (summary,) = summarize.summarize(tmp_path)
    assert summary["name"] == "b128-lr1e-3-s2x" and summary["seeds"] == [4]
    assert summary["main_score"]["std"] == 0.0


def test_configurations_are_ranked_by_challenge_score(tmp_path):
    for name, main_score, challenge in [("low", 0.9, 0.4), ("high", 0.1, 0.6)]:
        write_run(tmp_path, name, 0, {1: main_score}, epochs=1)
        write_scores(tmp_path / f"hydra-challenge-scores-{name}.json", {f"runs/hydra-{name}-s0/model.onnx": challenge})
    write_run(tmp_path, "unscored", 0, {1: 0.99}, epochs=1)
    summaries = summarize.summarize(tmp_path)
    assert [summary["name"] for summary in summaries] == ["high", "low", "unscored"]
    assert summaries[0]["challenge_score"]["mean"] == pytest.approx(0.6)


def test_curve_averages_every_evaluated_epoch_over_the_seeds(tmp_path):
    write_run(tmp_path, "e800", 0, {50: 0.4, 100: 0.6}, epochs=100)
    write_run(tmp_path, "e800", 1, {50: 0.5, 100: 0.8}, epochs=100)
    write_run(tmp_path, "other", 0, {50: 0.0, 100: 0.0}, epochs=100)
    write_scores(tmp_path / "hydra-challenge-curve-e800.json", {
        "runs/hydra-e800-s0/model-epoch0050.onnx": 0.3, "runs/hydra-e800-s1/model-epoch0050.onnx": 0.5,
        "runs/hydra-e800-s0/model-epoch0100.onnx": 0.7, "runs/hydra-e800-s1/model-epoch0100.onnx": 0.9})
    points = summarize.curve(tmp_path, "e800")
    assert list(points) == [50, 100]
    assert points[50]["main_score"]["mean"] == pytest.approx(0.45)
    assert points[100]["main_score"]["mean"] == pytest.approx(0.7)
    assert points[50]["challenge_score"]["mean"] == pytest.approx(0.4)
    assert points[100]["challenge_score"]["mean"] == pytest.approx(0.8)


def test_tables_print_missing_scores_as_dashes(tmp_path, capsys):
    write_run(tmp_path, "e800", 0, {1: 0.25}, epochs=1)
    summarize.print_summary(summarize.summarize(tmp_path))
    summarize.print_curve(summarize.curve(tmp_path, "e800"))
    lines = capsys.readouterr().out.splitlines()
    assert lines[1].split() == ["e800", "1", "0.250", "0.000", "0.250", "0.250", "-", "-", "-", "-"]
    assert lines[3].split() == ["1", "0.250", "0.000", "0.250", "0.250", "-", "-", "-", "-"]
