"""Params dataclass for the 2D synthetic ablation study (contact-distance variant)."""

from dataclasses import dataclass


@dataclass
class Params:
    """Hyperparameters for the 2D synthetic ablation study (contact-distance variant)."""

    # Network specs
    n_features = 384
    n_steps = 20
    n_layers = 4
    n_frequencies = 3
    t1 = 1.

    # Training specs
    batch_size = 700
    n_batches = 1
    n_iterations = 2500
    lr = 6e-7
    seed = 1
    amp_enabled = False

    # Loss specs
    entropy = 0.005
    scaling = 0.5
    w_b = 10
    w_m = 10
    w_g = 0.6
    w_mass = 10
    w_reg = 0.5
    w_limit = 0
    max_allowed = 4.

    # Experiment
    experiment = "ablation"
    dim = 2
    visualization_batch = 800
    dim_reduction = None

    # Intermediate marginals
    tm_idxs = [4]  # intermediate marginals to include in training
    tkm_idxs = [n_steps]  # which ResNet blocks the intermediate marginals correspond to

    # --- Ablation switches (variant a = ContactUOT baseline) ---
    use_jacobi_distance = True 
    logm_time_only = False  
    normalization_mode = "none"
