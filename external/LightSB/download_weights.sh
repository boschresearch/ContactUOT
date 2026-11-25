#!/bin/bash
# Downloads the ALAE FFHQ decoder checkpoint used by CuOT/decode_latents.py
# to decode ffhq latent bridges.
#
# Source: LightSB/ALAE/training_artifacts/download_all.py's own S3 fallback
# URL (https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth) --
# a plain HTTPS mirror, no auth or confirmation dance needed.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/ALAE/training_artifacts/ffhq"
DEST_FILE="$DEST_DIR/model_157.pth"
URL="https://alaeweights.s3.us-east-2.amazonaws.com/ffhq/model_157.pth"

if [ -f "$DEST_FILE" ]; then
    echo "Already present: $DEST_FILE"
    exit 0
fi

mkdir -p "$DEST_DIR"
echo "Downloading ALAE FFHQ decoder (~622MB) to $DEST_FILE ..."
# Write to .part and rename on success, so an interrupted download is never
# mistaken for a complete file by the "Already present" check.
curl -fL "$URL" -o "$DEST_FILE.part"
mv "$DEST_FILE.part" "$DEST_FILE"
echo "Done: $DEST_FILE"
