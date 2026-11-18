"""Checks for the diagnostics of trained runs: target entropy and prototype usage."""

import math

import numpy as np
import pytest
import torch

import main
import swav
import swav_data
import swav_diagnostics

ONE_DOMAIN_PER_CLASS = {0: "a", 1: "b", 2: "c"}  # the fixture has classes 0, 1 and 2


@pytest.fixture(scope="module")
def run_dir(eval_data, tmp_path_factory):
    """A one-epoch run on the synthetic data: 41 images, batch 8, 10 prototypes."""
    run_dir = tmp_path_factory.mktemp("diagnostics") / "run"
    main.main(["--data", str(eval_data), "--output", str(run_dir), "--device", "cpu", "--batch-size", "8",
               "--prototypes", "10", "--epochs", "1"])
    return run_dir


def test_prototype_usage_counts_used_and_effective_prototypes():
    even = swav_diagnostics.prototype_usage(np.array([0, 0, 3, 3]), num_prototypes=5)
    assert even == {"images": 4, "prototypes_used": 2, "effective_prototypes": pytest.approx(2.0)}
    uneven = swav_diagnostics.prototype_usage(np.array([0] * 9 + [3]), num_prototypes=5)
    assert uneven["prototypes_used"] == 2 and 1 < uneven["effective_prototypes"] < 2


def test_domain_usage_separates_shared_and_exclusive_prototypes():
    assignments = np.array([0, 1, 1, 2, 2, 3])
    domains = np.array(["a", "a", "a", "b", "b", "b"])
    domains[2] = "b"  # prototype 1 is now used by both domains
    usage = swav_diagnostics.domain_usage(assignments, domains, num_prototypes=5)
    assert usage["a"]["images"] == 2 and usage["b"]["images"] == 4
    assert usage["a"]["prototypes_used"] == 2 and usage["a"]["exclusive_prototypes"] == 1
    assert usage["b"]["prototypes_used"] == 3 and usage["b"]["exclusive_prototypes"] == 2


def test_target_entropy_follows_the_sharpness_of_sinkhorn(eval_data):
    torch.manual_seed(0)
    model = swav.SwAVModel(num_prototypes=10).eval()

    def entropy(epsilon):
        torch.manual_seed(0)
        loader = swav_data.make_train_loader(str(eval_data), batch_size=8, seed=0)
        return swav_diagnostics.target_entropy(model, loader, epsilon, 3, torch.device("cpu"))

    assert entropy(1e3) == pytest.approx(math.log(10), abs=1e-3)  # a huge epsilon gives uniform targets
    assert 0 <= entropy(0.01) < entropy(0.05) < math.log(10)


def test_loaded_run_has_the_trained_weights(run_dir):
    model, settings = swav_diagnostics.load_run(run_dir, torch.device("cpu"))
    trained = torch.load(run_dir / "checkpoint.pt")
    assert not model.training and settings["prototypes"] == 10
    assert all(torch.equal(weights, trained[name]) for name, weights in model.state_dict().items())


def test_diagnose_reports_consistent_numbers_and_is_repeatable(eval_data, run_dir):
    result = swav_diagnostics.diagnose(run_dir, eval_data, domain_of_class=ONE_DOMAIN_PER_CLASS)
    assert result["batch_size"] == 8 and result["prototypes"] == 10
    assert 0 <= result["target_entropy"] <= result["max_entropy"] == math.log(10)
    assert result["images"] == 41  # every image counts for the overall usage, labelled or not
    assert 1 <= result["effective_prototypes"] <= result["prototypes_used"] <= 10
    assert {domain: usage["images"] for domain, usage in result["per_domain"].items()} == {"a": 12, "b": 12, "c": 12}
    for usage in result["per_domain"].values():
        assert 0 <= usage["exclusive_prototypes"] <= usage["prototypes_used"] <= result["prototypes_used"]
    assert swav_diagnostics.diagnose(run_dir, eval_data, domain_of_class=ONE_DOMAIN_PER_CLASS) == result


def test_batch_size_can_be_replaced_for_the_targets(eval_data, run_dir):
    own = swav_diagnostics.diagnose(run_dir, eval_data, domain_of_class=ONE_DOMAIN_PER_CLASS)
    larger = swav_diagnostics.diagnose(run_dir, eval_data, batch_size=16, domain_of_class=ONE_DOMAIN_PER_CLASS)
    assert larger["batch_size"] == 16 and larger["target_entropy"] != own["target_entropy"]
    assert larger["prototypes_used"] == own["prototypes_used"]  # the usage does not depend on the batch size
