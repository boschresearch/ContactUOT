#!/bin/bash
# Downloads the UCF101 BreastStroke video latents used by the `ucf101_dc`
# (DC-AE f32c32 latents, dataset.UCF101Dataset) and `ucf101_titok`
# (TiTok-S-128 latents, dataset.UCF101TiTokDataset) experiments.
#
# Source: this project's Zenodo record, https://zenodo.org/records/22998248
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/ucf101"
RECORD_FILES="https://zenodo.org/records/22998248/files"
mkdir -p "$DEST_DIR"

download() {
    local name="$1"
    local dest="$DEST_DIR/$name"
    if [ -f "$dest" ]; then
        echo "Already present: $dest"
        return
    fi
    echo "Downloading $name to $dest ..."
    # Write to .part and rename on success, so an interrupted download is
    # never mistaken for a complete file by the "Already present" check.
    curl -fL "$RECORD_FILES/$name?download=1" -o "$dest.part"
    mv "$dest.part" "$dest"
}

download "ucf101_bs_latents_dc.pt"     # ~53MB, ucf101_dc
download "ucf101_bs_latents_titok.pt"  # ~40MB, ucf101_titok

echo "Done: $DEST_DIR"
