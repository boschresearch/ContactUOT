"""Params dataclass for the synthetic gene-expression simulation experiment."""

from dataclasses import dataclass


@dataclass
class Params:
    """Hyperparameters for the synthetic gene-expression simulation experiment."""

    # Network specs
    n_features = 512
    n_steps = 20
    n_layers = 3
    n_frequencies = 3
    t1 = 1.

    # Training specs
    batch_size = 512
    n_batches = 1
    n_iterations = 2500
    lr = 1e-5
    seed = 1
    amp_enabled = True

    # Loss specs
    entropy = 0.005
    scaling = 0.5
    w_b = 10
    w_m = 10
    w_g = 0.01
    w_mass = 10.
    w_reg = 0.05
    w_limit = 0
    max_allowed = 4.

    # Experiment
    experiment = "simulation"
    dim = 2
    visualization_batch = 400
    dim_reduction = None

    # Intermediate marginals
    tm_idxs = [1, 2, 3, 4]  # intermediate marginals to include in training
    tkm_idxs = [5, 10, 15, 20]  # which ResNet blocks the intermediate marginals correspond to
