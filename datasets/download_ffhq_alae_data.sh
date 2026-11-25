#!/bin/bash
# Downloads the ALAE-latent FFHQ data used by the `ffhq_alae` experiment
# (dataset.FFHQDataset) and shared with `ffhq_psp` (dataset.FFHQPSPDataset,
# for the gender/age category labels).
#
# Source: the LightSB paper's own notebook (notebooks/LightSB_alae.ipynb in
# the LightSB repository, https://arxiv.org/abs/2310.01174), whose
# "download pre-processed ALAE data" cell hosts these on Google Drive.
# Downloaded via `uvx gdown` (not a project dependency).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST_DIR="$SCRIPT_DIR/ffhq"
mkdir -p "$DEST_DIR"

download() {
    local name="$1" file_id="$2"
    local dest="$DEST_DIR/$name"
    if [ -f "$dest" ]; then
        echo "Already present: $dest"
        return
    fi
    echo "Downloading $name to $dest ..."
    # gdown downloads to its own temp .part file and only moves it to $dest
    # on success, so an interrupted download never leaves a partial $dest.
    uvx gdown "$file_id" -O "$dest"
}

download "age.npy" "1Vi6NzxCsS23GBNq48E-97Z9UuIuNaxPJ"
download "gender.npy" "1SEdsmQGL3mOok1CPTBEfc_O1750fGRtf"
download "latents.npy" "1ENhiTRsHtSjIjoRu1xYprcpNd8M9aVu8"

echo "Done: $DEST_DIR"
