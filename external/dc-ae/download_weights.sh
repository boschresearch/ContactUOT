#!/bin/bash
# Downloads the DC-AE (Chen et al., 2025, "Deep Compression Autoencoder for
# Efficient High-Resolution Diffusion Models", ICLR) f32c32 checkpoint used by
# CuOT/decode_latents.py to decode ucf101_dc latent bridges. The model code
# itself is diffusers' AutoencoderDC, so only the weights are needed here.
#
# Source: DC-AE's HuggingFace collection, linked from
# https://github.com/mit-han-lab/efficientvit -- repo
# mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers. Fetched as plain HTTPS from
# HuggingFace's resolve/ endpoint, pinned to a fixed commit.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/pretrained_models/dc-ae-f32c32-mix-1.0-diffusers"
REPO="mit-han-lab/dc-ae-f32c32-mix-1.0-diffusers"
REVISION="541db95f181d83ac0dc3c5f52e55c45c85c8ee4d"

if [ -f "$DEST_DIR/diffusion_pytorch_model.safetensors" ] && [ -f "$DEST_DIR/config.json" ]; then
    echo "Already present: $DEST_DIR"
    exit 0
fi

mkdir -p "$DEST_DIR"
echo "Downloading DC-AE f32c32 autoencoder (~1.3GB) to $DEST_DIR ..."
# Write to .part and rename on success, so an interrupted download is never
# mistaken for a complete file by the "Already present" check.
for FILE in config.json diffusion_pytorch_model.safetensors; do
    curl -fL "https://huggingface.co/$REPO/resolve/$REVISION/$FILE" -o "$DEST_DIR/$FILE.part"
    mv "$DEST_DIR/$FILE.part" "$DEST_DIR/$FILE"
done
echo "Done: $DEST_DIR"
