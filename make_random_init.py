"""Export an untrained ResNet9 as our score floor."""
import sys

import stable_pretraining as spt
import torch

from export_onnx import export_onnx

seed = int(sys.argv[1]) if len(sys.argv) > 1 else 0
torch.manual_seed(seed)

encoder = spt.backbone.Resnet9(1024, 3)
encoder.eval()
export_onnx(encoder, f"random_init_seed{seed}.onnx")
