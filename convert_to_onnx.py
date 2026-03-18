# -*- coding: utf-8 -*-
"""
Converts mass_best_model (3).pth -> model.onnx using the legacy ONNX exporter.

Usage:
    pip install torch torchvision onnx segmentation-models-pytorch
    python convert_to_onnx.py
"""

import sys
import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

# Force UTF-8 output so Windows cp1252 doesn't crash on unicode chars
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── Config ────────────────────────────────────────────────────────────────────

WEIGHTS_PATH = r"C:\Users\User\Downloads\pls work\mass_best_model (3).pth"
OUTPUT_PATH  = r"C:\Users\User\Downloads\pls work\model.onnx"
IMG_SIZE     = 320

# ── Model ─────────────────────────────────────────────────────────────────────

class SegmentationUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.unet = smp.UnetPlusPlus(
            encoder_name="efficientnet-b4",
            encoder_weights=None,      # no download needed, we load weights below
            in_channels=3,
            classes=1,
            activation=None,
            decoder_channels=(128, 64, 32, 16, 8),
            decoder_attention_type=None,
        )

    def forward(self, x):
        features    = self.unet.encoder(x)
        decoder_out = self.unet.decoder(features)
        seg_logit   = self.unet.segmentation_head(decoder_out)
        return seg_logit

# ── Load weights ──────────────────────────────────────────────────────────────

print("Loading model...")
model = SegmentationUNet()

state = torch.load(WEIGHTS_PATH, map_location="cpu")

# Unwrap common checkpoint wrappers
if isinstance(state, dict):
    for key in ("model_state_dict", "state_dict", "model"):
        if key in state:
            print(f"  Found checkpoint key '{key}', unwrapping...")
            state = state[key]
            break

model.load_state_dict(state)
model.eval()
print("Weights loaded OK")

# ── Export via LEGACY exporter (bypasses dynamo entirely) ─────────────────────

print(f"Exporting to {OUTPUT_PATH} ...")

dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

with torch.no_grad():
    torch.onnx.export(
        model,
        dummy,
        OUTPUT_PATH,
        export_params=True,
        opset_version=16,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input":  {0: "batch"},
            "output": {0: "batch"},
        },
        verbose=False,
    )

print("Done! model.onnx saved.")
print(f"  Input  shape: [1, 3, {IMG_SIZE}, {IMG_SIZE}]")
print(f"  Output shape: [1, 1, {IMG_SIZE}, {IMG_SIZE}]  (raw logits)")
print()
print("Next steps:")
print("  1. Upload model.onnx + index.html to your GitHub repo")
print("  2. If model.onnx > 100 MB run:  git lfs track \"*.onnx\"")