"""Params dataclass for the Veres pancreatic beta-cell differentiation experiment."""

from dataclasses import dataclass


@dataclass
class Params:
    """Hyperparameters for the Veres pancreatic beta-cell differentiation experiment."""

    # Network specs
    n_features = 512
    n_steps = 21
    n_layers = 4
    n_frequencies = 4
    t1 = 1.

    # Training specs
    batch_size = 1000
    n_batches = 12
    n_iterations = 500
    lr = 1e-5
    seed = 1
    amp_enabled = True

    # Loss specs
    entropy = 0.005
    scaling = 0.5
    w_b = 10
    w_m = 1
    w_g = 0.3
    w_mass = 1e2
    w_reg = 0.05
    w_limit = 1e4
    max_allowed = 4.

    # Experiment
    experiment = "veres"
    dim = 30
    visualization_batch = 750
    dim_reduction = "umap"

    # Intermediate marginals
    tm_idxs = [1, 2, 3, 4, 5, 6, 7]  # intermediate marginals to include in training
    tkm_idxs = [3, 6, 9, 12, 15, 18, 21]  # which ResNet blocks the intermediate marginals correspond to
