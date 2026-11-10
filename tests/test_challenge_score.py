"""Checks for the local stand-in of the leaderboard score."""

from collections import Counter

import numpy as np
from conftest import MODEL

import challenge_score


def test_every_class_belongs_to_one_of_three_domains_of_ten():
    assert sorted(challenge_score.DOMAIN_OF_CLASS) == list(range(30))
    assert Counter(challenge_score.DOMAIN_OF_CLASS.values()) == {
        challenge_score.IMAGENETTE: 10, challenge_score.GALAXY10: 10, challenge_score.EUROSAT: 10}


def test_in_domain_probes_are_fit_per_domain():
    # Class 0 (domain a) and class 2 (domain b) share their features exactly, so one probe over
    # both domains must confuse them; a probe per domain never sees them together.
    rng = np.random.default_rng(0)
    labels = np.repeat([0, 1, 2, 3], 20)
    domains = np.array(["a"] * 40 + ["b"] * 40)
    pattern = {0: 0, 1: 1, 2: 0, 3: 1}
    features = np.eye(2)[[pattern[label] for label in labels]] * 5 + rng.normal(0, 0.1, (len(labels), 2))
    assert challenge_score.in_domain_accuracy(features, labels, domains) == {"a": 1.0, "b": 1.0}


def test_few_shot_episode_has_exact_support_and_disjoint_queries():
    labels = np.repeat([0, 1, 2], [15, 12, 10])  # class 2 has no image left over for a query
    support, query = challenge_score.few_shot_split(labels, shots=10, rng=np.random.default_rng(0))
    assert Counter(labels[support]) == {0: 10, 1: 10}
    assert Counter(labels[query]) == {0: 5, 1: 2}
    assert not set(support) & set(query)


def test_few_shot_episodes_are_the_same_for_every_model(monkeypatch):
    labels, domains = np.repeat(np.arange(4), 15), np.array(["a"] * 60)
    first_model, second_model = np.random.default_rng(1).normal(size=(2, 60, 8))
    record = []
    original_split = challenge_score.few_shot_split

    def recording_split(labels, shots, rng):
        support, query = original_split(labels, shots, rng)
        record.append((support.tolist(), query.tolist()))
        return support, query

    monkeypatch.setattr(challenge_score, "few_shot_split", recording_split)
    challenge_score.few_shot_accuracy(first_model, labels, domains, episodes=3)
    challenge_score.few_shot_accuracy(second_model, labels, domains, episodes=3)
    assert record[:3] == record[3:]


def test_few_shot_accuracy_separates_clear_classes_and_not_noise():
    rng = np.random.default_rng(0)
    labels, domains = np.repeat(np.arange(5), 20), np.array(["a"] * 100)
    clear = np.eye(5)[labels] * 5 + rng.normal(0, 0.5, (100, 5))
    noise = rng.normal(size=(100, 5))
    assert challenge_score.few_shot_accuracy(clear, labels, domains)["a"] > 0.95
    assert challenge_score.few_shot_accuracy(noise, labels, domains)["a"] < 0.5


def test_score_of_a_real_model_is_the_mean_of_both_parts(eval_data):
    one_domain = {label: "only" for label in range(3)}  # the fixture has classes 0, 1 and 2
    result = challenge_score.challenge_score(MODEL, eval_data, domain_of_class=one_domain)
    assert set(result["per_domain"]) == {"only"}
    assert result["score"] == (result["in_domain"] + result["few_shot"]) / 2
    assert 0 <= result["few_shot"] <= 1 and 0 <= result["in_domain"] <= 1


def test_thread_count_follows_the_cpus_the_job_may_use(monkeypatch):
    monkeypatch.setattr(challenge_score.os, "sched_getaffinity", lambda pid: {0, 1, 2, 3}, raising=False)
    assert challenge_score.available_cpus() == 4
    monkeypatch.delattr(challenge_score.os, "sched_getaffinity", raising=False)
    assert challenge_score.available_cpus() == challenge_score.os.cpu_count()
