"""Params dataclass for the FFHQ face-latent interpolation experiment (ALAE latent space)."""

from dataclasses import dataclass


@dataclass
class Params:
    """Hyperparameters for the FFHQ face-latent interpolation experiment (ALAE latent space)."""

    # Network specs
    n_features = 512
    n_steps = 20
    n_layers = 4
    n_frequencies = 3
    t1 = 1.

    # Training specs
    batch_size = 1000
    n_batches = 5
    n_iterations = 10000
    lr = 1e-5
    seed = 1
    amp_enabled = True

    # Loss specs
    entropy = 0.005
    scaling = 0.5
    w_b = 1
    w_m = 1
    w_g = 0.1
    w_mass = 5
    w_reg = 0.1
    w_limit = 1e1
    max_allowed = 10.

    # Experiment
    experiment = "ffhq_alae"
    dim = 512
    visualization_batch = 10000
    dim_reduction = "umap"

    # Intermediate marginals
    tm_idxs = [1]  # intermediate marginals to include in training
    tkm_idxs = [20]  # which ResNet blocks the intermediate marginals correspond to
