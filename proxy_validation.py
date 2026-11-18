"""Record every leaderboard upload and check how well our local scores predict the leaderboard.

    uv run proxy_validation.py add MODEL.onnx --name b128-e200-s4 --date 2026-11-12 [--note TEXT]
    uv run proxy_validation.py leaderboard b128-e200-s4 0.7012
    uv run proxy_validation.py report

The leaderboard scores an upload about once a day, so we choose models with local scores instead:
``challenge_score.py`` (our stand-in for the leaderboard) and ``evaluate.py`` (the score ``main.py``
prints). ``uploads.json`` keeps one record per upload:

    add          hashes the uploaded file, computes both local scores and appends the record
    leaderboard  adds the official score once it is published
    report       prints all records and, over the uploads with an official score, the correlation of
                 each local score with it: Pearson (are the scores proportional?) and Spearman (do they
                 rank the models alike?)

Only the latest upload counts on the leaderboard, so an upload that was replaced before it was scored
keeps an empty leaderboard score. A correlation needs at least ``MIN_UPLOADS`` scored uploads, and with
so few points it is rough evidence. It is only meaningful if the uploads span a wide range of scores,
from the untrained encoder to the best model.
"""

import argparse
import hashlib
import json
import statistics
from pathlib import Path

UPLOADS_PATH = Path(__file__).resolve().parent / "uploads.json"
LOCAL_SCORES = ("challenge_score", "main_score")
MIN_UPLOADS = 3  # with two points every correlation is +1 or -1


def load_uploads(path):
    """All upload records, oldest first; an empty list if nothing was recorded yet."""
    return json.loads(Path(path).read_text()) if Path(path).is_file() else []


def save_uploads(path, uploads):
    Path(path).write_text(json.dumps(uploads, indent=2) + "\n")


def sha256_of_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_scores(model_path, data_path):
    """Both local scores of an ONNX model on the labelled training images."""
    # Imported here so that the report runs without loading the model libraries.
    import challenge_score
    import training

    return {"challenge_score": challenge_score.challenge_score(model_path, data_path)["score"],
            "main_score": training.probe_score(model_path, data_path)}


def add_upload(uploads, name, date, sha256, scores, note=""):
    """Append the record of one upload; its leaderboard score stays empty until it is published.

    Args:
        uploads: List of records, changed in place.
        name: Unique name of the upload, for example the configuration and seed.
        date: Day of the upload, as YYYY-MM-DD.
        sha256: Hash of the uploaded file.
        scores: Dict with the ``LOCAL_SCORES`` of the model.
        note: Optional remark, for example why the upload was replaced.
    """
    if any(upload["name"] == name for upload in uploads):
        raise ValueError(f"an upload named {name!r} is already recorded")
    uploads.append({"date": date, "name": name, "sha256": sha256,
                    **{score: round(scores[score], 4) for score in LOCAL_SCORES},
                    "leaderboard_score": None, "note": note})


def set_leaderboard_score(uploads, name, score):
    """Store the official score of the upload called ``name``."""
    for upload in uploads:
        if upload["name"] == name:
            upload["leaderboard_score"] = score
            return
    raise ValueError(f"no upload named {name!r}; recorded: {[upload['name'] for upload in uploads]}")


def ranks(values):
    """Rank of every value, 1 for the smallest; equal values share the mean of their ranks."""
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            result[order[position]] = (start + end) / 2 + 1
        start = end + 1
    return result


def correlations(uploads):
    """Correlation of each local score with the leaderboard score over the scored uploads.

    Returns:
        Dict from local score name to {"uploads": n, "pearson": r, "spearman": rho}. Both correlations
        are None with fewer than ``MIN_UPLOADS`` scored uploads or if one of the scores never varies.
    """
    scored = [upload for upload in uploads if upload["leaderboard_score"] is not None]
    official = [upload["leaderboard_score"] for upload in scored]
    result = {}
    for score in LOCAL_SCORES:
        local = [upload[score] for upload in scored]
        if len(scored) < MIN_UPLOADS or len(set(local)) == 1 or len(set(official)) == 1:
            result[score] = {"uploads": len(scored), "pearson": None, "spearman": None}
        else:
            result[score] = {"uploads": len(scored), "pearson": statistics.correlation(local, official),
                             "spearman": statistics.correlation(ranks(local), ranks(official))}
    return result


def print_report(uploads):
    def cell(value):
        return "-" if value is None else f"{value:.3f}"

    print(f"{'date':10} {'upload':28} {'sha256':12} {'challenge':>9} {'main':>6} {'leaderboard':>11}  note")
    for upload in uploads:
        print(f"{upload['date']:10} {upload['name']:28} {upload['sha256'][:12]} {cell(upload['challenge_score']):>9} "
              f"{cell(upload['main_score']):>6} {cell(upload['leaderboard_score']):>11}  {upload['note']}")
    print()
    for score, result in correlations(uploads).items():
        if result["pearson"] is None:
            print(f"{score}: {result['uploads']} scored uploads, "
                  f"a correlation needs {MIN_UPLOADS} with different scores")
        else:
            print(f"{score}: Pearson {result['pearson']:.3f}, Spearman {result['spearman']:.3f} "
                  f"over {result['uploads']} scored uploads")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--uploads", type=Path, default=UPLOADS_PATH, help="the file with the upload records")
    commands = parser.add_subparsers(dest="command", required=True)
    add = commands.add_parser("add", help="record an upload")
    add.add_argument("model", help="the uploaded ONNX file")
    add.add_argument("--name", required=True, help="unique name of the upload, e.g. b128-e200-s4")
    add.add_argument("--date", required=True, help="day of the upload, YYYY-MM-DD")
    add.add_argument("--note", default="")
    add.add_argument("--data", default="data/train.parquet", help="path to train.parquet")
    leaderboard = commands.add_parser("leaderboard", help="store the official score of an upload")
    leaderboard.add_argument("name")
    leaderboard.add_argument("score", type=float)
    commands.add_parser("report", help="print all uploads and the correlations")
    args = parser.parse_args()

    uploads = load_uploads(args.uploads)
    if args.command == "add":
        add_upload(uploads, args.name, args.date, sha256_of_file(args.model), local_scores(args.model, args.data),
                   args.note)
        save_uploads(args.uploads, uploads)
    elif args.command == "leaderboard":
        set_leaderboard_score(uploads, args.name, args.score)
        save_uploads(args.uploads, uploads)
    print_report(uploads)
