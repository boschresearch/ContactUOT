"""Params dataclass for the zebrafish embryogenesis experiment."""

from dataclasses import dataclass


@dataclass
class Params:
    """Hyperparameters for the zebrafish embryogenesis experiment."""

    # Network specs
    n_features = 512
    n_steps = 20
    n_layers = 4
    n_frequencies = 3
    t1 = 1.

    # Training specs
    batch_size = 1000
    n_batches = 7
    n_iterations = 500
    lr = 5e-4
    seed = 1
    amp_enabled = True

    # Loss specs
    entropy = 0.005
    scaling = 0.5
    w_b = 1
    w_m = 1
    w_g = 0.1
    w_mass = 1.
    w_reg = 0.1
    w_limit = 1e4
    max_allowed = 4.

    # Experiment
    experiment = "zebrafish"
    dim = 50
    visualization_batch = 2000
    dim_reduction = "umap"

    # Intermediate marginals
    tm_idxs = [1, 2, 3, 4, 5, 6]  # intermediate marginals to include in training
    tkm_idxs = [3, 7, 10, 14, 17, 20]  # which ResNet blocks the intermediate marginals correspond to
