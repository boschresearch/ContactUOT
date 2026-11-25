#!/bin/bash
# Downloads the TiTok-S-128 tokenizer checkpoint used by
# CuOT/decode_latents.py to decode ucf101_titok latent bridges.
#
# Source: 1d-tokenizer's own README_TiTok.md model table
# (https://github.com/bytedance/1d-tokenizer), "TiTok-S-128" entry, which
# points to the HuggingFace repo yucornetto/tokenizer_titok_s128_imagenet.
# Fetched as plain HTTPS from HuggingFace's resolve/ endpoint, pinned to the
# commit the ucf101_titok latents were encoded with.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/pretrained_models/tokenizer_titok_s128_imagenet"
REPO="yucornetto/tokenizer_titok_s128_imagenet"
REVISION="aa8740991cc9e5965e6dea04caad8905193fc24b"

if [ -f "$DEST_DIR/model.safetensors" ] && [ -f "$DEST_DIR/config.json" ]; then
    echo "Already present: $DEST_DIR"
    exit 0
fi

mkdir -p "$DEST_DIR"
echo "Downloading TiTok-S-128 tokenizer (~335MB) to $DEST_DIR ..."
# Write to .part and rename on success, so an interrupted download is never
# mistaken for a complete file by the "Already present" check.
for FILE in config.json model.safetensors; do
    curl -fL "https://huggingface.co/$REPO/resolve/$REVISION/$FILE" -o "$DEST_DIR/$FILE.part"
    mv "$DEST_DIR/$FILE.part" "$DEST_DIR/$FILE"
done
echo "Done: $DEST_DIR"
