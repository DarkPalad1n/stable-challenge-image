"""Checks for one training epoch: prototype freezing and the feature queue."""

import pytest
import torch

import swav
import swav_data
import training


@pytest.mark.parametrize("freeze_prototypes", [True, False])
def test_prototypes_move_only_when_not_frozen(eval_data, freeze_prototypes):
    torch.manual_seed(0)
    model = swav.SwAVModel(num_prototypes=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=0.05)
    loader = swav_data.make_train_loader(str(eval_data), batch_size=8, seed=0)
    prototypes_before = model.prototypes.weight.detach().clone()

    def loss_function(crop_scores, queue_scores):
        return swav.swav_loss(crop_scores, swav_data.NUM_GLOBAL_CROPS, 0.05, 3, 0.1, queue_scores)

    training.train_one_epoch(model, loader, optimizer, loss_function, torch.device("cpu"), freeze_prototypes)
    largest_change = (model.prototypes.weight - prototypes_before).abs().max().item()
    # Re-normalising unit vectors every step moves them by float32 rounding (~1e-8) only;
    # an optimizer step at this learning rate moves them by orders of magnitude more.
    if freeze_prototypes:
        assert largest_change < 1e-6
    else:
        assert largest_change > 1e-3


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

    training.train_one_epoch(model, loader, optimizer, recording_loss, torch.device("cpu"), True, queue)
    assert [queue_scores is None for _, queue_scores in steps] == [True, True, False, False, False]
    for step in range(2, len(steps)):
        queue_scores = steps[step][1]
        for crop_index in range(swav_data.NUM_GLOBAL_CROPS):
            expected = torch.cat([steps[step - 1][0][crop_index], steps[step - 2][0][crop_index]])
            torch.testing.assert_close(queue_scores[crop_index], expected)
