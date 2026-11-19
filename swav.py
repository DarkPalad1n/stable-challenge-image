"""SwAV (Caron et al., 2020): self-supervised learning by swapping cluster assignments.

Every image is seen as several crops (see ``swav_data.py``). Each crop is embedded by the
backbone, projected onto the unit sphere and compared with a set of learnable prototype
vectors. The training target of a global crop is its soft assignment to the prototypes,
computed with the Sinkhorn-Knopp algorithm so that each batch spreads evenly over all
prototypes. Every other crop of the same image must predict that target: the "swapped
prediction" that gives the method its name.

With small batches, a batch has too few crops to spread evenly over all prototypes.
``FeatureQueue`` then adds the projections of recent batches to Sinkhorn's input, as the
paper does for its small-batch runs.

Only ``SwAVModel.backbone`` is exported for the challenge. The projector, the prototypes
and the queue exist for the training objective alone.
"""

import math

import stable_pretraining as spt
import torch
import torch.nn.functional as F
from torch import nn

EMBEDDING_DIM = 1024  # the submission's embedding size; smaller backbone outputs are zero-padded on export
PROJECTOR_HIDDEN_DIM = 1024
PROJECTION_DIM = 128

BACKBONES = ("resnet9", "resnet18-narrow")
NARROW_RESNET18_WIDTHS = (40, 80, 160, 320)  # channels of the four stages
# stable-pretraining's Resnet9 uses this BatchNorm momentum, so the running statistics follow the last
# backbone pass (see SwAVModel.embed_crops). The narrow ResNet18 uses the same value.
BATCHNORM_MOMENTUM = 0.9


class BasicBlock(nn.Module):
    """ResNet basic block (He et al., 2016): two 3x3 convolutions plus a shortcut."""

    def __init__(self, in_channels, out_channels, stride):
        """Build the block.

        Args:
            in_channels: Channels of the input.
            out_channels: Channels of the output.
            stride: Stride of the first convolution; 2 halves the resolution.
        """
        super().__init__()
        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BATCHNORM_MOMENTUM),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=BATCHNORM_MOMENTUM),
        )
        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:  # a 1x1 convolution matches the shape of the residual branch
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels, momentum=BATCHNORM_MOMENTUM),
            )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.residual(x) + self.shortcut(x))


class NarrowResNet18(nn.Module):
    """The ResNet18 layout with narrower stages, for 96-pixel images and the 20 MiB cap.

    A standard ResNet18 (stages of 64-512 channels, 11.2M parameters, about 43 MiB) is over the
    cap, and its 7x7 stride-2 stem with max pooling shrinks a 96-pixel image to a quarter at once.
    This version keeps the four stages of two basic blocks each, with ``NARROW_RESNET18_WIDTHS``
    channels (4.37M parameters, about 16.7 MiB), starts with a 3x3 stride-1 convolution as
    ResNets for small images do, and ends with global average pooling. Weights start at random.
    """

    def __init__(self, widths=NARROW_RESNET18_WIDTHS):
        """Build the network.

        Args:
            widths: Channels of the four stages; the output has ``widths[-1]`` features.
        """
        super().__init__()
        layers = [nn.Conv2d(3, widths[0], 3, 1, padding=1, bias=False),
                  nn.BatchNorm2d(widths[0], momentum=BATCHNORM_MOMENTUM),
                  nn.ReLU()]
        in_channels = widths[0]
        for stage, width in enumerate(widths):
            layers += [BasicBlock(in_channels, width, stride=1 if stage == 0 else 2), BasicBlock(width, width, 1)]
            in_channels = width
        layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
        self.layers = nn.Sequential(*layers)
        self.output_dim = widths[-1]

    def forward(self, images):
        return self.layers(images)


def make_backbone(name):
    """Build a randomly initialised backbone and return it with its output size.

    Args:
        name: One of ``BACKBONES``: "resnet9" (stable-pretraining's ResNet9, 1024 features) or
            "resnet18-narrow" (``NarrowResNet18``, 320 features).
    """
    if name == "resnet9":
        return spt.backbone.Resnet9(EMBEDDING_DIM, 3), EMBEDDING_DIM
    if name == "resnet18-narrow":
        backbone = NarrowResNet18()
        return backbone, backbone.output_dim
    raise ValueError(f"unknown backbone {name!r}, expected one of {BACKBONES}")


class SwAVModel(nn.Module):
    """Backbone followed by a projection head and the prototypes."""

    def __init__(self, num_prototypes, backbone="resnet9"):
        """Build the model.

        Args:
            num_prototypes: Number of prototype vectors (the "clusters" crops are assigned to).
            backbone: Name of the backbone, see ``make_backbone``. It is built first, so with the
                same seed it starts from the same weights as ``make_random_init.py`` exports.
        """
        super().__init__()
        self.backbone, backbone_dim = make_backbone(backbone)
        self.projector = nn.Sequential(
            nn.Linear(backbone_dim, PROJECTOR_HIDDEN_DIM),
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

        The smallest resolution goes first. The backbones' BatchNorm layers use
        ``momentum=0.9``, so their running statistics, which the exported model uses,
        come almost entirely from the last pass. Ending on the global crops keeps those
        statistics at the 96-pixel resolution the evaluator uses.

        Args:
            crops: List of crop batches, each of shape [batch_size, 3, size, size].

        Returns:
            Tensor of shape [num_crops * batch_size, backbone output size], in the order of ``crops``.
        """
        embeddings_per_crop = [None] * len(crops)
        for resolution in sorted({crop.shape[-1] for crop in crops}):
            crop_indices = [index for index, crop in enumerate(crops) if crop.shape[-1] == resolution]
            embeddings = self.backbone(torch.cat([crops[index] for index in crop_indices]))
            for index, crop_embeddings in zip(crop_indices, embeddings.chunk(len(crop_indices))):
                embeddings_per_crop[index] = crop_embeddings
        return torch.cat(embeddings_per_crop)

    def forward(self, crops):
        """Project every crop onto the unit sphere and score it against every prototype.

        All crops go through the projector together, so its BatchNorm sees global and
        local crops alike, as in the original implementation.

        Args:
            crops: List of crop batches as yielded by ``swav_data.make_train_loader``.

        Returns:
            Two lists with one entry per crop, in the order of ``crops``:
            projections, each of shape [batch_size, PROJECTION_DIM] and of unit length;
            scores, each of shape [batch_size, num_prototypes]. Scores are cosine
            similarities, so they lie in [-1, 1].
        """
        batch_size = crops[0].shape[0]
        projections = F.normalize(self.projector(self.embed_crops(crops)), dim=1)
        crop_scores = self.prototypes(projections)
        return list(projections.split(batch_size)), list(crop_scores.split(batch_size))


class FeatureQueue:
    """Projections of the global crops from the most recent batches, newest first.

    Their scores are prepended to the batch's scores before Sinkhorn-Knopp, so the even
    spread over the prototypes is enforced over many more crops than one batch holds.
    The stored projections are re-scored with the current prototypes at every step;
    they never receive gradients.
    """

    def __init__(self, length, batch_size, num_global_crops, device):
        """Create an empty queue.

        Args:
            length: Number of projections kept per global crop. Must be a positive
                multiple of ``batch_size``, so whole batches enter and leave.
            batch_size: Number of images per training batch.
            num_global_crops: Number of leading crops whose projections are stored.
            device: Device the projections live on.
        """
        if length <= 0 or length % batch_size != 0:
            raise ValueError(f"queue length {length} must be a positive multiple of the batch size {batch_size}")
        self.length = length
        self.num_global_crops = num_global_crops
        self.projections = torch.zeros(num_global_crops, length, PROJECTION_DIM, device=device)
        self.num_stored = 0

    @property
    def is_full(self):
        """True once ``length`` projections are stored; only a full queue is used."""
        return self.num_stored == self.length

    @torch.no_grad()
    def scores(self, prototypes):
        """Score the stored projections against the current prototypes.

        Returns:
            One tensor of shape [length, num_prototypes] per global crop.
        """
        return [prototypes(projections) for projections in self.projections]

    @torch.no_grad()
    def push(self, crop_projections):
        """Store the global crops' projections of this batch and drop the oldest batch.

        Args:
            crop_projections: Projections per crop as returned by ``SwAVModel.forward``,
                global crops first.
        """
        newest = torch.stack(crop_projections[:self.num_global_crops])
        batch_size = newest.shape[1]
        self.projections = torch.cat([newest, self.projections[:, :-batch_size]], dim=1)
        self.num_stored = min(self.num_stored + batch_size, self.length)


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


def swav_loss(crop_scores, num_global_crops, epsilon, num_iterations, temperature, queue_scores=None):
    """SwAV's swapped-prediction loss (paper, eq. 1 and 2).

    Targets come from the global crops only; the paper found that targets from small crops
    hurt transfer. Each target is predicted by every other crop of the same image, global
    or local. With 2 global and 4 local crops this gives 2 x 5 = 10 cross-entropy terms,
    and the loss is their mean.

    Args:
        crop_scores: Scores per crop as returned by ``SwAVModel.forward``, global crops first.
        num_global_crops: How many leading entries of ``crop_scores`` are global crops.
        epsilon: Sinkhorn entropy regularisation.
        num_iterations: Sinkhorn normalisation rounds.
        temperature: Softmax temperature of the predictions.
        queue_scores: Optional output of ``FeatureQueue.scores``. If given, Sinkhorn runs
            on the queue's scores followed by the batch's, and only the batch's
            assignments are kept as targets.

    Returns:
        Scalar tensor.
    """
    batch_size = crop_scores[0].shape[0]
    cross_entropies = []
    for target_index in range(num_global_crops):
        sinkhorn_input = crop_scores[target_index]
        if queue_scores is not None:
            sinkhorn_input = torch.cat([queue_scores[target_index], sinkhorn_input])
        targets = sinkhorn(sinkhorn_input, epsilon, num_iterations)[-batch_size:]
        for predictor_index, scores in enumerate(crop_scores):
            if predictor_index == target_index:
                continue
            log_predictions = F.log_softmax(scores / temperature, dim=1)
            cross_entropies.append(-(targets * log_predictions).sum(dim=1).mean())
    return torch.stack(cross_entropies).mean()
