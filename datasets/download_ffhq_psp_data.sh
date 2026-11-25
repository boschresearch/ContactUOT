#!/bin/bash
# Downloads the data used by the `ffhq_psp` experiment (dataset.FFHQPSPDataset):
# psp_latent.npy (pSp W+ codes, ~2.58GB) from this project's Zenodo record,
# plus the gender.npy/age.npy category labels it shares with `ffhq_alae`.
#
# Source: https://zenodo.org/records/22998248 (psp_latent.npy); the labels
# come from download_ffhq_alae_data.sh (see its header comment).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/ffhq"
DEST_FILE="$DEST_DIR/psp_latent.npy"
URL="https://zenodo.org/records/22998248/files/psp_latent.npy?download=1"

# gender.npy/age.npy (that script also fetches ALAE latents.npy, ~143MB).
"$SCRIPT_DIR/download_ffhq_alae_data.sh"

if [ -f "$DEST_FILE" ]; then
    echo "Already present: $DEST_FILE"
    exit 0
fi

mkdir -p "$DEST_DIR"
echo "Downloading psp_latent.npy (~2.58GB) to $DEST_FILE ..."
# Write to .part and rename on success, so an interrupted download is never
# mistaken for a complete file by the "Already present" check.
curl -fL "$URL" -o "$DEST_FILE.part"
mv "$DEST_FILE.part" "$DEST_FILE"
echo "Done: $DEST_FILE"
