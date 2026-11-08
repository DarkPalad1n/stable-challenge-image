"""Checks for the SwAV core: Sinkhorn assignments, model outputs and the swapped-prediction loss."""

import pytest
import torch
import torch.nn.functional as F

import swav

BATCH_SIZE, NUM_PROTOTYPES = 8, 5
NUM_GLOBAL_CROPS, NUM_LOCAL_CROPS = 2, 4
EPSILON, NUM_ITERATIONS, TEMPERATURE = 0.05, 3, 0.1
SUBMISSION_SIZE_CAP_BYTES = 20 * 1024 * 1024


def random_scores(seed, batch_size=BATCH_SIZE, num_prototypes=NUM_PROTOTYPES):
    """Cosine-similarity-like scores in [-1, 1]."""
    return torch.rand(batch_size, num_prototypes, generator=torch.Generator().manual_seed(seed)) * 2 - 1


def reference_sinkhorn(scores, epsilon, num_iterations):
    """Single-process version of distributed_sinkhorn from facebookresearch/swav (main_swav.py)."""
    assignments = torch.exp(scores.double() / epsilon).t()  # [num_prototypes, batch_size]
    num_prototypes, batch_size = assignments.shape
    assignments /= assignments.sum()
    for _ in range(num_iterations):
        assignments /= assignments.sum(dim=1, keepdim=True)
        assignments /= num_prototypes
        assignments /= assignments.sum(dim=0, keepdim=True)
        assignments /= batch_size
    assignments *= batch_size
    return assignments.t()


def random_crops(batch_size=4):
    """Two 96-pixel global crops followed by four 48-pixel local crops."""
    generator = torch.Generator().manual_seed(0)
    global_crops = [torch.randn(batch_size, 3, 96, 96, generator=generator) for _ in range(NUM_GLOBAL_CROPS)]
    local_crops = [torch.randn(batch_size, 3, 48, 48, generator=generator) for _ in range(NUM_LOCAL_CROPS)]
    return global_crops + local_crops


def test_sinkhorn_rows_are_probability_distributions():
    assignments = swav.sinkhorn(random_scores(0), EPSILON, NUM_ITERATIONS)
    assert (assignments >= 0).all()
    torch.testing.assert_close(assignments.sum(dim=1), torch.ones(BATCH_SIZE))


def test_sinkhorn_balances_prototypes_when_converged():
    assignments = swav.sinkhorn(random_scores(0), EPSILON, num_iterations=500)
    expected_mass_per_prototype = torch.full((NUM_PROTOTYPES,), BATCH_SIZE / NUM_PROTOTYPES)
    torch.testing.assert_close(assignments.sum(dim=0), expected_mass_per_prototype, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("num_iterations", [1, 3, 10])
def test_sinkhorn_matches_original_swav_implementation(num_iterations):
    scores = random_scores(1)
    expected = reference_sinkhorn(scores, EPSILON, num_iterations).float()
    torch.testing.assert_close(swav.sinkhorn(scores, EPSILON, num_iterations), expected, rtol=1e-4, atol=1e-6)


def test_sinkhorn_stays_finite_for_tiny_epsilon():
    # exp(1 / 0.001) overflows float32; the log-space computation must not.
    assignments = swav.sinkhorn(random_scores(2), epsilon=0.001, num_iterations=NUM_ITERATIONS)
    assert torch.isfinite(assignments).all()


def test_sinkhorn_output_carries_no_gradient():
    scores = random_scores(3).requires_grad_()
    assert not swav.sinkhorn(scores, EPSILON, NUM_ITERATIONS).requires_grad


def test_model_returns_cosine_scores_for_every_crop():
    model = swav.SwAVModel(NUM_PROTOTYPES)
    crop_scores = model(random_crops(batch_size=4))
    assert [tuple(scores.shape) for scores in crop_scores] == [(4, NUM_PROTOTYPES)] * len(random_crops())
    assert all(scores.abs().max() <= 1 + 1e-5 for scores in crop_scores)


def test_backbone_sees_global_crops_last():
    model = swav.SwAVModel(NUM_PROTOTYPES)
    resolutions_seen = []
    model.backbone.register_forward_hook(lambda module, inputs, output: resolutions_seen.append(inputs[0].shape[-1]))
    model(random_crops())
    assert resolutions_seen == [48, 96]


def test_prototypes_have_unit_length_after_normalisation():
    model = swav.SwAVModel(NUM_PROTOTYPES)
    with torch.no_grad():
        model.prototypes.weight.mul_(3.0)
    model.normalize_prototypes()
    torch.testing.assert_close(model.prototypes.weight.norm(dim=1), torch.ones(NUM_PROTOTYPES))


def test_backbone_fits_the_submission_format():
    backbone = swav.SwAVModel(NUM_PROTOTYPES).backbone.eval()
    float32_size_bytes = 4 * sum(parameter.numel() for parameter in backbone.parameters())
    assert float32_size_bytes < SUBMISSION_SIZE_CAP_BYTES
    assert backbone(torch.randn(2, 3, 96, 96)).shape == (2, swav.EMBEDDING_DIM)


def test_loss_takes_targets_from_global_crops_only(monkeypatch):
    crop_scores = [random_scores(seed) for seed in range(NUM_GLOBAL_CROPS + NUM_LOCAL_CROPS)]
    target_sources = []
    real_sinkhorn = swav.sinkhorn

    def recording_sinkhorn(scores, epsilon, num_iterations):
        target_sources.append(next(index for index, candidate in enumerate(crop_scores) if candidate is scores))
        return real_sinkhorn(scores, epsilon, num_iterations)

    monkeypatch.setattr(swav, "sinkhorn", recording_sinkhorn)
    swav.swav_loss(crop_scores, NUM_GLOBAL_CROPS, EPSILON, NUM_ITERATIONS, TEMPERATURE)
    assert target_sources == [0, 1]


def test_loss_is_the_mean_over_all_ten_target_predictor_pairs():
    crop_scores = [random_scores(seed) for seed in range(NUM_GLOBAL_CROPS + NUM_LOCAL_CROPS)]
    target_predictor_pairs = [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5), (1, 0), (1, 2), (1, 3), (1, 4), (1, 5)]

    def cross_entropy(target_index, predictor_index):
        targets = swav.sinkhorn(crop_scores[target_index], EPSILON, NUM_ITERATIONS)
        log_predictions = F.log_softmax(crop_scores[predictor_index] / TEMPERATURE, dim=1)
        return -(targets * log_predictions).sum(dim=1).mean()

    expected = torch.stack([cross_entropy(*pair) for pair in target_predictor_pairs]).mean()
    loss = swav.swav_loss(crop_scores, NUM_GLOBAL_CROPS, EPSILON, NUM_ITERATIONS, TEMPERATURE)
    torch.testing.assert_close(loss, expected)


def test_loss_trains_backbone_projector_and_prototypes():
    model = swav.SwAVModel(NUM_PROTOTYPES)
    loss = swav.swav_loss(model(random_crops()), NUM_GLOBAL_CROPS, EPSILON, NUM_ITERATIONS, TEMPERATURE)
    loss.backward()
    assert torch.isfinite(loss)
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name
