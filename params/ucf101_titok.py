from dataclasses import dataclass
import torch


@dataclass
class Params:

    # Network specs
    # TiTok-S-128 (Yu et al., 2024): 128 latent tokens x 12 dims, no 2D grid
    # structure (vs. ucf101_dc.py's 32x8x8 spatial latent from Chen et al., 2025).
    # n_features is the vector-field MLP's hidden width, unrelated to the
    # data dim (`dim` below) -- kept at 2048 to match ucf101_dc.py's model size.
    n_features = 2048 # 4096
    n_steps = 171 # 101
    n_layers = 3 # 3
    n_frequencies = 5 # 5
    t1 = 2.

    # Training specs
    batch_size = 101 # 100
    n_batches = 1 # 1
    n_iterations = 15000 # 500 60000
    lr = 3e-5 # 4e-5
    seed = 1 # 1
    amp_enabled = False

    # Loss specs
    entropy = 0.005 #0.005
    scaling = 0.5
    w_b = 5
    w_m = 1
    w_g = 0.01 # 0.1
    w_mass = 5e2
    w_reg = 0.1
    w_limit = 5e4
    max_allowed = 4.

    # Experiment
    experiment = "ucf101_titok"
    dim = 1536
    visualization_batch = 100
    dim_reduction = "umap"

    # Intermediate marginals
    # intermediate marginals to include in training
    tm_idxs = list(range(1, 35))
    # which ResNet blocks the intermediate marginals correspond to
    tkm_idxs = list(range(5, 171, 5))
