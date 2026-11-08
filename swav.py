"""SwAV (Caron et al., 2020): self-supervised learning by swapping cluster assignments.

Every image is seen as several crops (see ``swav_data.py``). Each crop is embedded by the
backbone, projected onto the unit sphere and compared with a set of learnable prototype
vectors. The training target of a global crop is its soft assignment to the prototypes,
computed with the Sinkhorn-Knopp algorithm so that each batch spreads evenly over all
prototypes. Every other crop of the same image must predict that target: the "swapped
prediction" that gives the method its name.

Only ``SwAVModel.backbone`` is exported for the challenge. The projector and the
prototypes exist for the training objective alone.
"""

import math

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from torch import nn

EMBEDDING_DIM = 1024  # backbone output size, equal to the submission's embedding size
PROJECTOR_HIDDEN_DIM = 1024
PROJECTION_DIM = 128


class SwAVModel(nn.Module):
    """ResNet9 backbone followed by a projection head and the prototypes."""

    def __init__(self, num_prototypes):
        """Build the model.

        Args:
            num_prototypes: Number of prototype vectors (the "clusters" crops are assigned to).
        """
        super().__init__()
        self.backbone = spt.backbone.Resnet9(EMBEDDING_DIM, 3)
        self.projector = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, PROJECTOR_HIDDEN_DIM),
            nn.BatchNorm1d(PROJECTOR_HIDDEN_DIM),
            nn.ReLU(),
            nn.Linear(PROJECTOR_HIDDEN_DIM, PROJECTION_DIM),
        )
        self.prototypes = nn.Linear(PROJECTION_DIM, num_prototypes, bias=False)
        self.normalize_prototypes()

    @torch.no_grad()
    def normalize_prototypes(self):
        """Rescale every prototype to unit length. Call this before each training step."""
        self.prototypes.weight.copy_(F.normalize(self.prototypes.weight, dim=1))

    def embed_crops(self, crops):
        """Embed all crops with the backbone, one backbone pass per crop resolution.

        The smallest resolution goes first. The ResNet9's BatchNorm layers use
        ``momentum=0.9``, so their running statistics, which the exported model uses,
        come almost entirely from the last pass. Ending on the global crops keeps those
        statistics at the 96-pixel resolution the evaluator uses.

        Args:
            crops: List of crop batches, each of shape [batch_size, 3, size, size].

        Returns:
            Tensor of shape [num_crops * batch_size, EMBEDDING_DIM], in the order of ``crops``.
        """
        embeddings_per_crop = [None] * len(crops)
        for resolution in sorted({crop.shape[-1] for crop in crops}):
            crop_indices = [index for index, crop in enumerate(crops) if crop.shape[-1] == resolution]
            embeddings = self.backbone(torch.cat([crops[index] for index in crop_indices]))
            for index, crop_embeddings in zip(crop_indices, embeddings.chunk(len(crop_indices))):
                embeddings_per_crop[index] = crop_embeddings
        return torch.cat(embeddings_per_crop)

    def forward(self, crops):
        """Score every crop against every prototype.

        All crops go through the projector together, so its BatchNorm sees global and
        local crops alike, as in the original implementation.

        Args:
            crops: List of crop batches as yielded by ``swav_data.make_train_loader``.

        Returns:
            One tensor of shape [batch_size, num_prototypes] per crop, in the order of
            ``crops``. Entries are cosine similarities, so they lie in [-1, 1].
        """
        batch_size = crops[0].shape[0]
        projections = F.normalize(self.projector(self.embed_crops(crops)), dim=1)
        return list(self.prototypes(projections).split(batch_size))


@torch.no_grad()
def sinkhorn(scores, epsilon, num_iterations):
    """Turn prototype scores into soft assignments that use all prototypes equally.

    Solves SwAV's entropy-regularised transport problem (paper, eq. 3) with the
    Sinkhorn-Knopp algorithm. The computation runs in log space, so a small ``epsilon``
    cannot overflow ``exp``. No gradient flows through the result: it is a target.

    Args:
        scores: Tensor of shape [batch_size, num_prototypes], crop-prototype similarities.
        epsilon: Entropy regularisation; smaller values give sharper assignments.
        num_iterations: Number of normalisation rounds (3 in the paper).

    Returns:
        Tensor of shape [batch_size, num_prototypes]. Each row is a probability
        distribution over the prototypes, and the columns are balanced: every prototype
        receives about batch_size / num_prototypes of the mass (exactly in the limit).
    """
    batch_size, num_prototypes = scores.shape
    log_assignments = scores.float() / epsilon
    for _ in range(num_iterations):
        # Give every prototype the same total mass, 1 / num_prototypes ...
        log_assignments -= torch.logsumexp(log_assignments, dim=0, keepdim=True) + math.log(num_prototypes)
        # ... and every crop the same total mass, 1 / batch_size.
        log_assignments -= torch.logsumexp(log_assignments, dim=1, keepdim=True) + math.log(batch_size)
    return torch.exp(log_assignments + math.log(batch_size))


def swav_loss(crop_scores, num_global_crops, epsilon, num_iterations, temperature):
    """SwAV's swapped-prediction loss (paper, eq. 1 and 2).

    Targets come from the global crops only; the paper found that targets from small crops
    hurt transfer. Each target is predicted by every other crop of the same image, global
    or local. With 2 global and 4 local crops this gives 2 x 5 = 10 cross-entropy terms,
    and the loss is their mean.

    Args:
        crop_scores: Output of ``SwAVModel.forward``, global crops first.
        num_global_crops: How many leading entries of ``crop_scores`` are global crops.
        epsilon: Sinkhorn entropy regularisation.
        num_iterations: Sinkhorn normalisation rounds.
        temperature: Softmax temperature of the predictions.

    Returns:
        Scalar tensor.
    """
    cross_entropies = []
    for target_index in range(num_global_crops):
        targets = sinkhorn(crop_scores[target_index], epsilon, num_iterations)
        for predictor_index, scores in enumerate(crop_scores):
            if predictor_index == target_index:
                continue
            log_predictions = F.log_softmax(scores / temperature, dim=1)
            cross_entropies.append(-(targets * log_predictions).sum(dim=1).mean())
    return torch.stack(cross_entropies).mean()
