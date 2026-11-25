"""Decode x0.npy / x_pred.npy latent-bridge outputs from the `ffhq_alae`,
`ffhq_psp`, `ucf101_dc` and `ucf101_titok` experiments back into images.

Uses the vendored decoders under `external/`:
  - `external/LightSB/ALAE`            -- ALAE decoder, for `ffhq_alae`
    (dim=512 ALAE W-space codes, see dataset.FFHQDataset).
  - `external/pixel2style2pixel`       -- pSp's StyleGAN2 decoder, for
    `ffhq_psp` (dim=9216 W+ codes, see dataset.FFHQPSPDataset).
  - `external/dc-ae`                   -- DC-AE f32c32 weights for diffusers'
    AutoencoderDC, for `ucf101_dc` (dim=2048=32x8x8 codes, see
    dataset.UCF101Dataset).
  - `external/1d-tokenizer`            -- TiTok-S-128 decoder, for
    `ucf101_titok` (dim=1536=12x1x128 codes, see dataset.UCF101TiTokDataset).

Only the decoder half of each autoencoder is vendored/used here -- encoding
is never needed since x0.npy/x_pred.npy already are latent codes.

CLI usage (writes a transition_grid.png next to the input .npy files):
    uv run python CuOT/decode_latents.py --experiment ffhq_psp --dir results/figures_ffhq_psp_date_20260727_011310
"""
import argparse
import contextlib
import logging
import os
import sys

import numpy as np
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ALAE_DIR = os.path.join(_REPO_ROOT, "external", "LightSB", "ALAE")
_PSP_DIR = os.path.join(_REPO_ROOT, "external", "pixel2style2pixel")
_DCAE_DIR = os.path.join(
    _REPO_ROOT, "external", "dc-ae", "pretrained_models", "dc-ae-f32c32-mix-1.0-diffusers"
)
_TITOK_DIR = os.path.join(_REPO_ROOT, "external", "1d-tokenizer")

sys.path.insert(0, _REPO_ROOT)
from dataset import FFHQ_DATA_DIR, compute_psp_latent_stats  # noqa: E402

_alae_model = None
_psp_decoder = None
_psp_mean_std = None
_dcae_model = None
_titok_model = None


def _default_device():
    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


@contextlib.contextmanager
def _torch_default_device(device):
    """ALAE's net.py creates its per-pixel noise via bare `torch.randn(...)`
    calls (no explicit device=), relying on the process-global default
    device -- the original ALAE scripts set this once via
    `torch.set_default_device("cuda")` at import time. Scope that same
    default here instead of leaving it changed process-wide."""
    previous = torch.get_default_device()
    torch.set_default_device(device)
    try:
        yield
    finally:
        torch.set_default_device(previous)


def _load_alae_model(device):
    """Build the ALAE FFHQ model and load `model_157.pth`, matching
    LightSB/ALAE/alae_ffhq_inference.py::load_model, but with absolute paths
    so this works regardless of the caller's cwd."""
    global _alae_model
    if _alae_model is not None:
        return _alae_model

    # ALAE's own modules (model.py, net.py, ...) import each other by bare
    # name (e.g. `from registry import *`), not as a package -- they need
    # their own directory on sys.path, not just the repo root.
    sys.path.insert(0, _ALAE_DIR)
    from model import Model
    from checkpointer import Checkpointer
    from defaults import get_cfg_defaults
    import lreq

    lreq.use_implicit_lreq.set(True)

    cfg = get_cfg_defaults()
    cfg.merge_from_file(os.path.join(_ALAE_DIR, "configs", "ffhq.yaml"))
    # Checkpointer.load() resolves the checkpoint via a hardcoded
    # "OUTPUT_DIR.split('/')[:-3]" hack that only works when OUTPUT_DIR is
    # this short *relative* path and cwd is _ALAE_DIR (see the chdir below)
    # -- an absolute OUTPUT_DIR strips the wrong path segments.
    cfg.OUTPUT_DIR = os.path.join("training_artifacts", "ffhq")
    cfg.freeze()

    model = Model(
        startf=cfg.MODEL.START_CHANNEL_COUNT,
        layer_count=cfg.MODEL.LAYER_COUNT,
        maxf=cfg.MODEL.MAX_CHANNEL_COUNT,
        latent_size=cfg.MODEL.LATENT_SPACE_SIZE,
        truncation_psi=cfg.MODEL.TRUNCATIOM_PSI,
        truncation_cutoff=cfg.MODEL.TRUNCATIOM_CUTOFF,
        mapping_layers=cfg.MODEL.MAPPING_LAYERS,
        channels=cfg.MODEL.CHANNELS,
        generator=cfg.MODEL.GENERATOR,
        encoder=cfg.MODEL.ENCODER,
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    model_dict = {
        "discriminator_s": model.encoder,
        "generator_s": model.decoder,
        "mapping_tl_s": model.mapping_d,
        "mapping_fl_s": model.mapping_f,
        "dlatent_avg": model.dlatent_avg,
    }
    checkpointer = Checkpointer(
        cfg, model_dict, {}, logger=logging.getLogger("logger"), save=False
    )
    _prev_cwd = os.getcwd()
    try:
        os.chdir(_ALAE_DIR)
        checkpointer.load()
    finally:
        os.chdir(_prev_cwd)
    model.eval()

    _alae_model = model
    return model


def decode_ffhq_alae(latents, device=None):
    """latents: (N, 512) ALAE W-space codes (dataset.FFHQDataset's space).

    Returns: (N, 3, 1024, 1024) uint8 tensor in [0, 255].
    """
    device = device or _default_device()
    model = _load_alae_model(device)
    latents = torch.as_tensor(np.asarray(latents), dtype=torch.float32, device=device)

    layer_count = 9
    images = []
    with torch.no_grad(), _torch_default_device(device):
        for i in range(latents.shape[0]):
            w = latents[i:i + 1, None, :].repeat(1, model.mapping_f.num_layers, 1)
            images.append(model.decoder(w, layer_count - 1, 1, noise=True))
    images = torch.cat(images)
    return ((images * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)


def _load_psp_decoder(device):
    """Load only the StyleGAN2 decoder half of the pSp checkpoint -- the
    encoder is never needed since inputs here are already W+ codes."""
    global _psp_decoder
    if _psp_decoder is not None:
        return _psp_decoder

    sys.path.insert(0, _PSP_DIR)
    from models.stylegan2.model import Generator

    ckpt_path = os.path.join(_PSP_DIR, "pretrained_models", "psp_ffhq_encode.pt")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"]
    decoder_sd = {
        k[len("decoder."):]: v for k, v in state_dict.items() if k.startswith("decoder.")
    }

    generator = Generator(1024, 512, 8)
    generator.load_state_dict(decoder_sd, strict=True)
    generator.to(device)
    generator.eval()
    generator.requires_grad_(False)

    _psp_decoder = generator
    return generator


def _psp_latent_stats():
    """Recompute the same per-dimension mean/std FFHQPSPDataset standardizes
    with (dataset.compute_psp_latent_stats), so it can be inverted."""
    global _psp_mean_std
    if _psp_mean_std is not None:
        return _psp_mean_std
    raw = np.load(os.path.join(FFHQ_DATA_DIR, "psp_latent.npy")).reshape(-1, 18 * 512)
    _psp_mean_std = compute_psp_latent_stats(raw)
    del raw
    return _psp_mean_std


def decode_ffhq_psp(latents_flat, device=None, standardized=True):
    """latents_flat: (N, 9216) pSp W+ codes, flattened.

    Args:
        standardized: True if latents are in FFHQPSPDataset's standardized
            space (the space x0.npy/x_pred.npy are saved in) -- inverts the
            same per-dimension transform before decoding. False if latents
            are already raw pSp W+ codes.

    Returns: (N, 3, 1024, 1024) uint8 tensor in [0, 255].
    """
    device = device or _default_device()
    decoder = _load_psp_decoder(device)

    latents_flat = np.asarray(latents_flat, dtype=np.float32)
    if standardized:
        mean, std = _psp_latent_stats()
        latents_flat = latents_flat * std + mean

    codes = torch.as_tensor(latents_flat, dtype=torch.float32, device=device).view(-1, 18, 512)
    with torch.no_grad():
        images, _ = decoder(
            [codes], input_is_latent=True, randomize_noise=False, return_latents=True
        )
    return ((images * 0.5 + 0.5) * 255).clamp(0, 255).to(torch.uint8)


def _load_dcae_model(device):
    """Load the DC-AE f32c32 autoencoder from the weights downloaded by
    external/dc-ae/download_weights.sh."""
    global _dcae_model
    if _dcae_model is not None:
        return _dcae_model

    from diffusers.models import AutoencoderDC

    model = AutoencoderDC.from_pretrained(_DCAE_DIR)
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    _dcae_model = model
    return model


def decode_ucf101_dc(latents_flat, device=None, batch_size=16):
    """latents_flat: (N, 2048) DC-AE codes, flattened from (32, 8, 8) and
    multiplied by the model's scaling_factor at encoding time
    (dataset.UCF101Dataset's space).

    Returns: (N, 3, 256, 256) uint8 tensor in [0, 255].
    """
    device = device or _default_device()
    model = _load_dcae_model(device)
    codes = torch.as_tensor(np.asarray(latents_flat), dtype=torch.float32).view(-1, 32, 8, 8)

    images = []
    with torch.no_grad():
        for start in range(0, codes.shape[0], batch_size):
            chunk = codes[start:start + batch_size].to(device)
            images.append(model.decode(chunk / model.config.scaling_factor).sample.cpu())
    images = torch.cat(images)
    return ((images.clamp(-1, 1) * 0.5 + 0.5) * 255).to(torch.uint8)


def _load_titok_model(device):
    """Load the TiTok-S-128 tokenizer from the weights downloaded by
    external/1d-tokenizer/download_weights.sh."""
    global _titok_model
    if _titok_model is not None:
        return _titok_model

    # 1d-tokenizer's modules import each other as the top-level `modeling`
    # package, so its root directory needs to be on sys.path.
    sys.path.insert(0, _TITOK_DIR)
    from modeling.titok import TiTok

    model = TiTok.from_pretrained(
        os.path.join(_TITOK_DIR, "pretrained_models", "tokenizer_titok_s128_imagenet")
    )
    model.to(device)
    model.eval()
    model.requires_grad_(False)

    _titok_model = model
    return model


def decode_ucf101_titok(latents_flat, device=None, batch_size=32):
    """latents_flat: (N, 1536) TiTok-S-128 quantized codes, flattened from
    (12, 1, 128) (dataset.UCF101TiTokDataset's space).

    Returns: (N, 3, 256, 256) uint8 tensor in [0, 255].
    """
    device = device or _default_device()
    model = _load_titok_model(device)
    codes = torch.as_tensor(np.asarray(latents_flat), dtype=torch.float32).view(-1, 12, 1, 128)

    images = []
    with torch.no_grad():
        for start in range(0, codes.shape[0], batch_size):
            chunk = codes[start:start + batch_size].to(device)
            images.append(model.decode(chunk).cpu())
    images = torch.cat(images)
    return (images.clamp(0, 1) * 255).to(torch.uint8)


DECODERS = {
    "ffhq_alae": decode_ffhq_alae,
    "ffhq_psp": decode_ffhq_psp,
    "ucf101_dc": decode_ucf101_dc,
    "ucf101_titok": decode_ucf101_titok,
}


def _transition_grid(x_pred, decode_fn, num_samples=10, num_steps=5, sample_inds=None):
    """Build a (num_steps, num_samples) grid of decoded images across the
    trajectory in x_pred, matching bridge_visualizer[.py|_psp.py]'s layout."""
    if sample_inds is None:
        sample_inds = tuple(range(min(num_samples, x_pred.shape[1])))
    time_inds = np.linspace(0, x_pred.shape[0] - 1, num_steps).astype(int)

    selected = x_pred[time_inds][:, list(sample_inds)]          # (steps, samples, dim)
    selected = selected.transpose(1, 0, 2)                      # (samples, steps, dim)

    decoded_rows = []
    for i in tqdm(range(selected.shape[0]), desc="decoding"):
        imgs = decode_fn(selected[i])
        decoded_rows.append(imgs.cpu().permute(0, 2, 3, 1).numpy())
    return np.array(decoded_rows)  # (samples, steps, H, W, 3)


def _save_grid(decoded, save_path):
    num_samples, num_steps = decoded.shape[:2]
    fig, axes = plt.subplots(
        num_steps, num_samples, figsize=(num_samples * 2, num_steps * 2), dpi=200
    )
    for i in range(num_samples):
        for k in range(num_steps):
            axes[k, i].imshow(decoded[i, k])
            axes[k, i].axis("off")
    plt.tight_layout(pad=0.05)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved", save_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=list(DECODERS), required=True)
    parser.add_argument(
        "--dir", required=True,
        help="Directory containing x0.npy/x_pred.npy -- the "
             "results/figures_<experiment>_date_.../ folder for the run you want "
             "decoded (main.py saves them there alongside that run's other results).",
    )
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--num-steps", type=int, default=5)
    args = parser.parse_args()

    x_pred = np.load(os.path.join(args.dir, "x_pred.npy"))
    print("x_pred:", x_pred.shape)

    decoded = _transition_grid(x_pred, DECODERS[args.experiment], args.num_samples, args.num_steps)
    _save_grid(decoded, os.path.join(args.dir, f"transition_grid_{args.experiment}.png"))


if __name__ == "__main__":
    main()
