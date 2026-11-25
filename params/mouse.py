"""Params dataclass for the mouse hematopoiesis experiment."""

from dataclasses import dataclass


@dataclass
class Params:
    """Hyperparameters for the mouse hematopoiesis experiment."""

    # Network specs
    n_features = 512
    n_steps = 20
    n_layers = 4
    n_frequencies = 3
    t1 = 1.

    # Training specs
    batch_size = 512
    n_batches = 1
    n_iterations = 1500
    lr = 1e-5
    seed = 1
    amp_enabled = True

    # Loss specs
    entropy = 0.005
    scaling = 0.5
    w_b = 10
    w_m = 10
    w_g = 0.01
    w_mass = 1.
    w_reg = 0.05
    w_limit = 0
    max_allowed = 4.

    # Experiment
    experiment = "mouse"
    dim = 2
    visualization_batch = 500
    dim_reduction = None

    # Intermediate marginals
    tm_idxs = [1, 2]  # intermediate marginals to include in training
    tkm_idxs = [10, 20]  # which ResNet blocks the intermediate marginals correspond to
