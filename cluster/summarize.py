"""Summarise the runs of ``cluster/train_seeds.sh``: one row per configuration, or one training curve.

    uv run cluster/summarize.py [--runs runs]                  every configuration, last epoch
    uv run cluster/summarize.py [--runs runs] --curve NAME     one configuration, every evaluated epoch

This is the protocol behind the tables in ``SWAV-SUMMARY.md``. A configuration is the set of run
directories ``hydra-NAME-sSEED``. Only the last epoch of a run is reported, never its best one: picking
the best epoch would select the model with labels. Runs that have not finished are left out.

Two scores are shown as mean, standard deviation, minimum and maximum over the seeds:

    main.py score     the probe score of the last epoch in the run's ``history.json``
    challenge score   from ``hydra-challenge-scores-NAME.json``, written by ``cluster/score_models.sh``

The standard deviation is the sample one (n - 1), as used for the standard errors in the summary.
``--curve`` averages the same two scores over the seeds for every evaluated epoch; its challenge
scores come from ``hydra-challenge-curve-NAME.json`` (``score_models.sh NAME curve``).
"""

import argparse
import json
import re
import statistics
from pathlib import Path

RUN_DIR_PATTERN = re.compile(r"hydra-(?P<name>.+)-s(?P<seed>\d+)")
KEPT_MODEL_PATTERN = re.compile(r"model-epoch(?P<epoch>\d+)\.onnx")  # written by main.py --keep-evaluated-models


def load_json(path):
    return json.loads(Path(path).read_text())


def find_runs(runs_dir):
    """Group the finished runs under ``runs_dir`` by configuration.

    A run is finished when its history reaches the number of epochs in its ``config.json``.

    Returns:
        Dict from configuration name to a dict from seed to that run's history (list of epoch records).
    """
    configurations = {}
    for run_dir in sorted(Path(runs_dir).iterdir()):
        match = RUN_DIR_PATTERN.fullmatch(run_dir.name)
        if not match or not (run_dir / "history.json").is_file():
            continue
        history = load_json(run_dir / "history.json")
        if history and history[-1]["epoch"] == load_json(run_dir / "config.json")["settings"]["epochs"]:
            configurations.setdefault(match["name"], {})[int(match["seed"])] = history
    return configurations


def seed_of_model(model_path):
    """The seed in a model path such as ``runs/hydra-NAME-s3/model.onnx``."""
    return int(RUN_DIR_PATTERN.fullmatch(Path(model_path).parent.name)["seed"])


def final_challenge_scores(runs_dir, name):
    """Challenge score of every seed's final model, or an empty dict if the configuration is not scored yet.

    Returns:
        Dict from seed to score.
    """
    path = Path(runs_dir) / f"hydra-challenge-scores-{name}.json"
    if not path.is_file():
        return {}
    return {seed_of_model(model_path): result["score"] for model_path, result in load_json(path).items()}


def describe(values):
    """Mean, sample standard deviation, minimum and maximum of ``values``; None for no values.

    The standard deviation of a single value is reported as 0.
    """
    if not values:
        return None
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return {"mean": statistics.fmean(values), "std": deviation, "min": min(values), "max": max(values)}


def summarize(runs_dir):
    """One summary per configuration, best challenge score first (configurations without one last).

    Returns:
        List of dicts with "name", "seeds" (sorted list), "main_score" and "challenge_score"; the two
        scores are ``describe`` results, and "challenge_score" is None if the configuration is not scored.
    """
    summaries = []
    for name, histories in find_runs(runs_dir).items():
        summaries.append({
            "name": name,
            "seeds": sorted(histories),
            "main_score": describe([history[-1]["score"] for history in histories.values()]),
            "challenge_score": describe(list(final_challenge_scores(runs_dir, name).values())),
        })

    def rank(summary):
        challenge = summary["challenge_score"]
        return (challenge is None, -(challenge or summary["main_score"])["mean"])

    return sorted(summaries, key=rank)


def curve(runs_dir, name):
    """Both scores of one configuration for every evaluated epoch, averaged over the seeds.

    Returns:
        Dict from epoch to a dict with "main_score" and "challenge_score". Each is a ``describe`` result
        over the seeds that have this epoch, or None if no seed has it.
    """
    main_scores, challenge_scores = {}, {}
    for history in find_runs(runs_dir).get(name, {}).values():
        for record in history:
            if "score" in record:
                main_scores.setdefault(record["epoch"], []).append(record["score"])
    curve_path = Path(runs_dir) / f"hydra-challenge-curve-{name}.json"
    if curve_path.is_file():
        for model_path, result in load_json(curve_path).items():
            epoch = int(KEPT_MODEL_PATTERN.fullmatch(Path(model_path).name)["epoch"])
            challenge_scores.setdefault(epoch, []).append(result["score"])
    return {epoch: {"main_score": describe(main_scores.get(epoch, [])),
                    "challenge_score": describe(challenge_scores.get(epoch, []))}
            for epoch in sorted(set(main_scores) | set(challenge_scores))}


SCORE_COLUMNS = ("mean", "std", "min", "max")


def format_score(label, description=None, header=False):
    """One score's four columns behind its label: the column names, the values, or dashes if missing."""
    if header:
        cells = [f"{column:>6}" for column in SCORE_COLUMNS]
    elif description is None:
        cells = [f"{'-':>6}"] * len(SCORE_COLUMNS)
    else:
        cells = [f"{description[column]:6.3f}" for column in SCORE_COLUMNS]
    return f"{label if header else '':11}" + " ".join(cells)


def format_scores(point, header=False):
    """The main.py score and the challenge score of one table row, or the header above them."""
    return (format_score("main.py:", point and point["main_score"], header) + "  "
            + format_score("challenge:", point and point["challenge_score"], header))


def print_summary(summaries):
    print(f"{'configuration':24} seeds  {format_scores(None, header=True)}")
    for summary in summaries:
        print(f"{summary['name']:24} {len(summary['seeds']):5}  {format_scores(summary)}")


def print_curve(points):
    print(f"epoch  {format_scores(None, header=True)}")
    for epoch, point in points.items():
        print(f"{epoch:5}  {format_scores(point)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", default="runs", help="directory that holds the hydra-NAME-sSEED run directories")
    parser.add_argument("--curve", metavar="NAME", help="print the training curve of this configuration instead")
    args = parser.parse_args()
    if args.curve:
        print_curve(curve(args.runs, args.curve))
    else:
        print_summary(summarize(args.runs))
