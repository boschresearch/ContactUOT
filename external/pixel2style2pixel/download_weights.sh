#!/bin/bash
# Downloads the pSp FFHQ ("StyleGAN Inversion") decoder checkpoint used by
# CuOT/decode_latents.py to decode ffhq_psp latent bridges.
#
# Source: pixel2style2pixel's own README pretrained-models table
# (https://github.com/eladrich/pixel2style2pixel), "StyleGAN Inversion" entry,
# Google Drive file id 1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0. Downloaded via
# `uvx gdown` (not a project dependency) since Drive requires its
# large-file-confirmation handling.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/pretrained_models"
DEST_FILE="$DEST_DIR/psp_ffhq_encode.pt"
FILE_ID="1bMTNWkh5LArlaWSc_wa8VKyq2V42T2z0"

if [ -f "$DEST_FILE" ]; then
    echo "Already present: $DEST_FILE"
    exit 0
fi

mkdir -p "$DEST_DIR"
echo "Downloading pSp FFHQ decoder (~1.2GB) to $DEST_FILE ..."
uvx gdown "$FILE_ID" -O "$DEST_FILE"
echo "Done: $DEST_FILE"
