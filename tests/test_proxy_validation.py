"""Checks for the record of leaderboard uploads and the correlation of local and official scores."""

import hashlib

import pytest
from conftest import MODEL

import evaluate
import proxy_validation

SCORES = {"challenge_score": 0.61234, "main_score": 0.62549}


def uploads_with_scores(*triples):
    """Records with the given (challenge score, main.py score, leaderboard score) triples."""
    uploads = []
    for index, (challenge, main_score, official) in enumerate(triples):
        proxy_validation.add_upload(uploads, f"upload-{index}", "2026-11-12", "0" * 64,
                                    {"challenge_score": challenge, "main_score": main_score})
        if official is not None:
            proxy_validation.set_leaderboard_score(uploads, f"upload-{index}", official)
    return uploads


def test_upload_is_recorded_without_a_leaderboard_score_and_survives_saving(tmp_path):
    uploads = []
    proxy_validation.add_upload(uploads, "b128-e200-s4", "2026-11-12", "ab" * 32, SCORES, note="first")
    assert uploads == [{"date": "2026-11-12", "name": "b128-e200-s4", "sha256": "ab" * 32, "challenge_score": 0.6123,
                        "main_score": 0.6255, "leaderboard_score": None, "note": "first"}]
    proxy_validation.save_uploads(tmp_path / "uploads.json", uploads)
    assert proxy_validation.load_uploads(tmp_path / "uploads.json") == uploads
    assert proxy_validation.load_uploads(tmp_path / "missing.json") == []


def test_upload_names_are_unique():
    uploads = uploads_with_scores((0.5, 0.5, None))
    with pytest.raises(ValueError, match="already recorded"):
        proxy_validation.add_upload(uploads, "upload-0", "2026-11-13", "1" * 64, SCORES)


def test_leaderboard_score_is_stored_for_the_named_upload_only():
    uploads = uploads_with_scores((0.5, 0.5, None), (0.6, 0.6, None))
    proxy_validation.set_leaderboard_score(uploads, "upload-1", 0.7)
    assert [upload["leaderboard_score"] for upload in uploads] == [None, 0.7]
    with pytest.raises(ValueError, match="no upload named"):
        proxy_validation.set_leaderboard_score(uploads, "unknown", 0.7)


def test_file_hash_is_its_sha256(tmp_path):
    (tmp_path / "model.onnx").write_bytes(b"not a real model")
    assert proxy_validation.sha256_of_file(tmp_path / "model.onnx") == hashlib.sha256(b"not a real model").hexdigest()


def test_ranks_share_the_mean_rank_among_equal_values():
    assert proxy_validation.ranks([0.3, 0.1, 0.3, 0.9]) == [2.5, 1.0, 2.5, 4.0]


def test_correlations_use_only_scored_uploads_and_need_three():
    two_scored = uploads_with_scores((0.4, 0.4, 0.3), (0.5, 0.5, 0.6), (0.9, 0.9, None))
    assert proxy_validation.correlations(two_scored)["challenge_score"] == {
        "uploads": 2, "pearson": None, "spearman": None}


def test_pearson_measures_proportionality_and_spearman_the_ranking():
    # The challenge score ranks the uploads like the leaderboard but not in proportion; the main.py
    # score ranks them in reverse.
    uploads = uploads_with_scores((0.40, 0.9, 0.10), (0.41, 0.8, 0.20), (0.42, 0.7, 0.30), (0.90, 0.1, 0.31))
    result = proxy_validation.correlations(uploads)
    assert result["challenge_score"]["spearman"] == pytest.approx(1.0)
    assert 0 < result["challenge_score"]["pearson"] < 0.9
    assert result["main_score"]["spearman"] == pytest.approx(-1.0)
    assert result["main_score"]["uploads"] == 4


def test_constant_scores_have_no_correlation():
    uploads = uploads_with_scores((0.5, 0.4, 0.3), (0.5, 0.5, 0.4), (0.5, 0.6, 0.5))
    result = proxy_validation.correlations(uploads)
    assert result["challenge_score"]["pearson"] is None and result["main_score"]["pearson"] == pytest.approx(1.0)


def test_local_scores_are_the_two_scores_we_report(eval_data):
    scores = proxy_validation.local_scores(MODEL, eval_data)
    assert set(scores) == set(proxy_validation.LOCAL_SCORES)
    assert scores["main_score"] == evaluate.evaluate(str(MODEL), str(eval_data))["score"]
    assert 0 <= scores["challenge_score"] <= 1


def test_report_shows_missing_scores_as_dashes(capsys):
    proxy_validation.print_report(uploads_with_scores((0.5, 0.6, None)))
    output = capsys.readouterr().out
    assert "upload-0" in output and " -" in output
    assert "a correlation needs 3" in output
