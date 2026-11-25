# ContactUOT
Code repository of [Contact Geometry for Generative Models: An Unbalanced Optimal Transport Formulation](https://openreview.net/forum?id=E75sBSUY4i#discussion) (NeurIPS 26 poster).

Neural transport maps for unbalanced optimal transport, fit as geodesics of a
contact-Hamiltonian vector field that jointly models particle drift and mass
growth/decay. A `NeuralMap` is trained to interpolate between a sequence of
observed marginals (e.g. single-cell trajectories over time, or paired image
latents) while reproducing their relative population sizes.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
uv sync
```

## Usage

Training and evaluation are driven by a `params/<name>.py` module selected via `--params`:

```bash
uv run python CuOT/main.py --params simulation
```

This trains a model on the corresponding dataset (see
`CuOT/experiment_selector.py` for the name -> dataset mapping), saves a
checkpoint under `models/`, and writes trajectory/growth/quantitative-metric
plots under `results/`.

One example command per available experiment (see `params/`):

```bash
uv run python CuOT/main.py --params simulation
uv run python CuOT/main.py --params mouse
uv run python CuOT/main.py --params emt
uv run python CuOT/main.py --params veres
uv run python CuOT/main.py --params zebrafish
uv run python CuOT/main.py --params ffhq_alae
uv run python CuOT/main.py --params ffhq_psp
uv run python CuOT/main.py --params ucf101_dc
uv run python CuOT/main.py --params ucf101_titok
uv run python CuOT/main.py --params ablation
```

Other flags:

```bash
# Skip training, only run evaluation/plots on an existing checkpoint
uv run python CuOT/main.py --params simulation --model_name 20260926_020123

# Resume interrupted training from a saved checkpoint
uv run python CuOT/main.py --params veres --resume models/veres_date_20260926_130337_checkpoint.pth

# Override the params dataclass's default (e.g., seed)
uv run python CuOT/main.py --params mouse --seed 0
```

## Repository layout

```
ContactUOT/
├── CuOT/                       # core library + CLI entry point
│   ├── main.py                 # training/evaluation entry point
│   ├── dataset.py              # dataset classes (per-timestep particle batches)
│   ├── experiment_selector.py  # experiment name -> dataset/highlight-times mapping
│   ├── visualization.py        # trajectory/growth/timestep plotting
│   ├── quantitative_metrics.py # Sinkhorn-based quantitative evaluation
│   ├── utils.py                # metric logging and misc helpers
│   └── decode_latents.py       # decode ffhq_alae/ffhq_psp/ucf101_dc/ucf101_titok latent bridges back into images
├── networks/
│   ├── architectures.py        # the NeuralMap model and vector-field modules
│   └── training.py             # training loop and loss
├── params/                     # per-experiment hyperparameter dataclasses
│   ├── simulation.py
│   ├── mouse.py
│   ├── emt.py
│   ├── veres.py
│   ├── zebrafish.py
│   ├── ffhq_alae.py
│   ├── ffhq_psp.py
│   ├── ucf101_dc.py
│   ├── ucf101_titok.py
│   └── ablation.py
├── datasets/                   # CSV data, generate_ablation.py, download_*_data.sh for latent data
├── external/
│   ├── LightSB/ALAE/             # Vendored ALAE decoder (from baselines/LightSB), used to
│   │                             # decode `ffhq_alae` (dim=512 ALAE latents) trajectories.
│   ├── pixel2style2pixel/        # Vendored pSp StyleGAN2 decoder, used to decode `ffhq_psp`
│   │                             # (dim=9216 W+ latents) trajectories.
│   ├── dc-ae/                    # Vendored DC-AE, used to decode `ucf101_dc` (dim=2048) trajectories.
│   └── 1d-tokenizer/             # Vendored TiTok-S-128 decoder (from bytedance/1d-tokenizer), used
│                                 # to decode `ucf101_titok` (dim=1536) trajectories
├── models/                     # generated checkpoints (git-ignored)
├── results/                    # generated figures (git-ignored)
```

## Decoding latent bridges into images

`main.py` saves every run's final trajectory as `x0.npy`/`x_pred.npy` inside
that run's `results/figures_<experiment>_date_.../` folder (raw latent codes,
not images). For the `ffhq_alae`, `ffhq_psp`, `ucf101_dc` and `ucf101_titok`
experiments, those codes can be decoded back into images using the decoders
under `external/`:

- `external/LightSB/ALAE` (ALAE, for `ffhq_alae` face images)
- `external/pixel2style2pixel` (pSp's StyleGAN2 decoder, for `ffhq_psp` face images)
- `external/dc-ae` (DC-AE f32c32, for `ucf101_dc` video frames). The model code
  is `diffusers`' `AutoencoderDC`, so only the weights are fetched.
- `external/1d-tokenizer` (TiTok-S-128, for `ucf101_titok` video frames)

Only the decoder half of each autoencoder is used — encoding is never needed,
since `x0.npy`/`x_pred.npy` are already latent codes.

### Download the pretrained decoder weights

The decoder checkpoints are not committed to this
repository. Run each project's download script:

```bash
./external/LightSB/download_weights.sh          # ALAE FFHQ decoder, ~622MB
./external/pixel2style2pixel/download_weights.sh  # pSp FFHQ decoder, ~1.2GB
./external/dc-ae/download_weights.sh            # DC-AE f32c32 autoencoder, ~1.3GB
./external/1d-tokenizer/download_weights.sh     # TiTok-S-128 tokenizer, ~335MB
```

See each script's header comment for where its URL/file-id comes from (all
sourced from the original projects' own docs/scripts), and each directory's
`LICENSE` for its terms.

### Run the decoders

```bash
# ffhq_alae (ALAE, dim=512) -- writes transition_grid_ffhq_alae.png next to x_pred.npy
uv run python CuOT/decode_latents.py --experiment ffhq_alae --dir results/figures_ffhq_alae_date_20260926_122147

# ffhq_psp (pSp StyleGAN2, dim=9216 W+) -- writes transition_grid_ffhq_psp.png
uv run python CuOT/decode_latents.py --experiment ffhq_psp --dir results/figures_ffhq_psp_date_20260727_011310

# ucf101_dc (DC-AE f32c32, dim=2048=32x8x8) -- writes transition_grid_ucf101_dc.png
uv run python CuOT/decode_latents.py --experiment ucf101_dc --dir results/figures_ucf101_dc_date_<timestamp>

# ucf101_titok (TiTok-S-128, dim=1536=12x1x128) -- writes transition_grid_ucf101_titok.png
uv run python CuOT/decode_latents.py --experiment ucf101_titok --dir results/figures_ucf101_titok_date_<timestamp>
```

## Data

The CSV files under `datasets/` are tracked in this repository. The synthetic
ablation dataset (`datasets/ablation.csv`) can be regenerated with:

```bash
uv run python datasets/generate_ablation.py
```

### Image and video latent data

The latent datasets for `ffhq_alae`, `ffhq_psp`, `ucf101_dc` and `ucf101_titok`
are not committed to this repository. Run the download script for the
experiment(s) you want:

```bash
./datasets/download_ffhq_alae_data.sh  # ffhq_alae: ALAE latents + gender/age labels, ~145MB
./datasets/download_ffhq_psp_data.sh   # ffhq_psp: pSp W+ latents, ~2.58GB (also fetches the labels)
./datasets/download_ucf101_data.sh     # ucf101_dc + ucf101_titok: DC-AE and TiTok latents, ~93MB
```

The `ffhq_psp` and UCF101 latents are hosted on this project's
[Zenodo record](https://zenodo.org/records/22998248); the ALAE latents and
FFHQ gender/age labels come from the [LightSB paper](https://arxiv.org/abs/2310.01174)
authors' Google Drive. See each script's header comment for provenance.
