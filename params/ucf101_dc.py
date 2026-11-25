"""Params dataclass for the UCF101 video-latent interpolation experiment (DC-AE latent space)."""

from dataclasses import dataclass


@dataclass
class Params:
    """Hyperparameters for the UCF101 video-latent interpolation experiment (DC-AE latent space)."""

    # Network specs
    n_features = 2048
    n_steps = 171
    n_layers = 3
    n_frequencies = 5
    t1 = 2.

    # Training specs
    batch_size = 101
    n_batches = 1
    n_iterations = 15000
    lr = 3e-5
    seed = 1
    amp_enabled = False

    # Loss specs
    entropy = 0.005
    scaling = 0.5
    w_b = 5
    w_m = 1
    w_g = 0.01
    w_mass = 5e2
    w_reg = 0.1
    w_limit = 5e4
    max_allowed = 4.

    # Experiment
    experiment = "ucf101_dc"
    dim = 2048
    visualization_batch = 100
    dim_reduction = "umap"

    # Intermediate marginals
    tm_idxs = list(range(1, 35))  # intermediate marginals to include in training
    tkm_idxs = list(range(5, 171, 5))  # which ResNet blocks the intermediate marginals correspond to
