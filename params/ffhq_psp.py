"""Params dataclass for the FFHQ experiment on pSp-encoder W+ latents."""

from dataclasses import dataclass


@dataclass
class Params:
    """Hyperparameters for the FFHQ experiment on pSp-encoder W+ latents."""

    # Network specs
    n_features = 2048
    n_steps = 20
    n_layers = 3
    n_frequencies = 3
    t1 = 1.

    # Training specs
    batch_size = 500
    n_batches = 15
    n_iterations = 5000
    lr = 1e-5
    seed = 1
    amp_enabled = True

    # Loss specs
    entropy = 0.005
    scaling = 0.5
    w_b = 1
    w_m = 1
    w_g = 0.01
    w_mass = 0.1
    w_reg = 0.1
    w_limit = 1e1
    max_allowed = 4.

    # Experiment
    experiment = "ffhq_psp"
    dim = 18 * 512  # pSp W+ latent: 18 StyleGAN layers x 512-d style code
    # main.py runs the post-training eval/visualization forward pass on visualization_batch
    # samples in one shot, with no no_grad()/autocast, so it holds full fp32 activations for
    # all n_steps. At dim=9216 this is kept small to fit on a GPU shared with another job.
    visualization_batch = 100
    dim_reduction = "umap"

    # Intermediate marginals
    tm_idxs = [1]  # intermediate marginals to include in training
    tkm_idxs = [20]  # which ResNet blocks the intermediate marginals correspond to
