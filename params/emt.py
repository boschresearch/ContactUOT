"""Params dataclass for the EMT (epithelial-mesenchymal transition) experiment."""

from dataclasses import dataclass


@dataclass
class Params:
    """Hyperparameters for the EMT (epithelial-mesenchymal transition) experiment."""

    # Network specs
    n_features = 256
    n_steps = 21
    n_layers = 3
    n_frequencies = 3
    t1 = 1.

    # Training specs
    batch_size = 512
    n_batches = 4
    n_iterations = 2500
    lr = 1e-5
    seed = 1
    amp_enabled = True

    # Loss specs
    entropy = 0.005
    scaling = 0.5
    w_b = 10
    w_m = 10
    w_g = 0.3
    w_mass = 1
    w_reg = 0.05
    w_limit = 1e3
    max_allowed = 4.

    # Experiment
    experiment = "emt"
    dim = 10
    visualization_batch = 700
    dim_reduction = "raw"

    # Intermediate marginals
    tm_idxs = [1, 2, 3]  # intermediate marginals to include in training
    tkm_idxs = [7, 14, 20]  # which ResNet blocks the intermediate marginals correspond to
