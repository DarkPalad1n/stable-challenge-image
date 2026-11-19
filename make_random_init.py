"""Export an untrained backbone as our score floor.

    uv run make_random_init.py [SEED] [BACKBONE]

SEED defaults to 0 and BACKBONE to resnet9 (see ``swav.BACKBONES``). The weights are those a SwAV run
with the same seed and backbone starts from. Writes random_init_seedSEED.onnx for the ResNet9 and
random_init_BACKBONE_seedSEED.onnx for any other backbone.
"""
import sys

import torch

import swav
from export_onnx import export_onnx

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
backbone_name = sys.argv[2] if len(sys.argv) > 2 else "resnet9"
torch.manual_seed(seed)

encoder, _ = swav.make_backbone(backbone_name)
encoder.eval()
prefix = "random_init" if backbone_name == "resnet9" else f"random_init_{backbone_name}"
export_onnx(encoder, f"{prefix}_seed{seed}.onnx")
