"""
Generates the synthetic 2D "ablation" dataset used to compare ContactUOT
variants (see params/ablation*.py).

Storyline (5 time steps, indices 0..4):
  t=0: a single isotropic Gaussian centered at (0, 0), split (invisibly, since
       all three branches share the same distribution at t=0) into three
       branches of 200 samples each -- 600 samples total.
  t=4: three well-separated, shaped modes:
         - "up"    branch, swiss-roll shape, centered at ( 0.0,  2.0), 600 samples (grows 200 -> 600)
         - "left"  branch, rings shape,      centered at (-1.5, -1.0), 300 samples (grows 200 -> 300)
         - "right" branch, pinwheel shape,   centered at ( 1.5, -1.0), 100 samples (shrinks 200 -> 100)
  t=1,2,3: bridge snapshots that interpolate each branch's position from the
       shared origin Gaussian towards its final shape.

  Each branch has its OWN growth/death curve (see BRANCH_COUNTS) rather than
  a shared, constant fraction of a single growth curve -- the true mass
  dynamics are therefore genuinely a function of *where* a particle is, not
  just of time. This is what makes the "growth head as a function of time
  only" ablation (variant c) meaningfully harder than the full space+time
  variant: a time-only rate cannot simultaneously grow "up" and shrink
  "right" at the same instant.

Output: datasets/ablation.csv with columns [samples, x1, x2], where
"samples" holds the time index (0.0 .. 4.0), matching the convention used by
SimulationDataset / EmtDataset in dataset.py.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.makedirs("datasets", exist_ok=True)

SEED = 0
OUT_CSV = "datasets/ablation.csv"
OUT_PNG = "dataset_ablation.png"

# Branch counts per step (5 steps: t=0..4)
BRANCH_COUNTS = {
    "up":    [200, 240, 370, 290, 300],
    "left":  [200, 240, 270, 290, 300],
    "right": [200, 160, 130, 110, 100],
}

# Final target configurations (at t=4)
BRANCHES = {
    "up":    dict(center=np.array([0.0, 1.4]),    n_max=max(BRANCH_COUNTS["up"]),    shape="swiss_roll", noise=0.03),
    "left":  dict(center=np.array([-1.2, -0.7]), n_max=max(BRANCH_COUNTS["left"]),  shape="rings",      noise=0.03),
    "right": dict(center=np.array([1.2, -0.7]),  n_max=max(BRANCH_COUNTS["right"]), shape="pinwheel",   noise=0.02),
}

# Per-branch initial Gaussians (at t=0): offset toward center (0,0), non-overlapping
SOURCE_CONFIGS = {
    "up":    dict(center=np.array([0.0, 0.70]),   std=0.21),
    "left":  dict(center=np.array([-0.6, -0.35]), std=0.21),
    "right": dict(center=np.array([0.6, -0.35]), std=0.21),
}

N_STEPS = 5
BRIDGE_SIGMA = 0.15
SHAPE_SCALE = 0.9
ALPHAS = [0, 0.7, 0.85, 0.95, 1]


def sample_swiss_roll(rng, n, noise=0.03):
    """Sample n points on a 2D swiss-roll spiral, normalized to unit max extent.

    Args:
        rng: NumPy random Generator.
        n: Number of points.
        noise: Std of additive Gaussian noise.

    Returns:
        np.ndarray: Points, shape (n, 2).
    """
    u = rng.uniform(0.0, 1.0, size=n)
    t = 0.5 * np.pi + 4.0 * np.pi * u
    x = t * np.cos(t)
    y = t * np.sin(t)
    pts = np.stack([x, y], axis=1)
    pts = pts / (np.abs(pts).max() + 1e-8)
    if noise > 0:
        pts = pts + noise * rng.standard_normal(pts.shape)
    return pts


def sample_rings(rng, n, noise=0.03):
    """Sample n points split between two concentric rings.

    Args:
        rng: NumPy random Generator.
        n: Number of points.
        noise: Std of additive Gaussian noise.

    Returns:
        np.ndarray: Points, shape (n, 2).
    """
    ring_idx = rng.integers(0, 2, size=n)
    radius = np.where(ring_idx == 0, 0.4, 0.9)
    theta = rng.uniform(0.0, 2.0 * np.pi, size=n)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    pts = np.stack([x, y], axis=1)
    if noise > 0:
        pts = pts + noise * rng.standard_normal(pts.shape)
    return pts


def sample_pinwheel(rng, n, noise=0.02):
    """Sample n points on a 5-armed pinwheel pattern, normalized to unit max extent.

    Args:
        rng: NumPy random Generator.
        n: Number of points.
        noise: Std of additive Gaussian noise.

    Returns:
        np.ndarray: Points, shape (n, 2).
    """
    num_arms = 5
    arm = rng.integers(0, num_arms, size=n)
    r = rng.uniform(0.0, 1.0, size=n) * 2.0 + 0.2
    theta = arm * (2.0 * np.pi / num_arms) + 0.5 * r
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    pts = np.stack([x, y], axis=1)
    pts = pts / (np.abs(pts).max() + 1e-8)
    if noise > 0:
        pts = pts + noise * rng.standard_normal(pts.shape)
    return pts


def sample_shape(rng, n, spec):
    """Sample n points of the shape named in spec["shape"], recentered and scaled per spec.

    Args:
        rng: NumPy random Generator.
        n: Number of points.
        spec: Dict with keys "shape", "center", "noise" (see BRANCHES).

    Returns:
        np.ndarray: Points, shape (n, 2).

    Raises:
        ValueError: If spec["shape"] is not swiss_roll, rings, or pinwheel.
    """
    if spec["shape"] == "swiss_roll":
        raw = sample_swiss_roll(rng, n, noise=spec["noise"])
    elif spec["shape"] == "rings":
        raw = sample_rings(rng, n, noise=spec["noise"])
    elif spec["shape"] == "pinwheel":
        raw = sample_pinwheel(rng, n, noise=spec["noise"])
    else:
        raise ValueError(f"Unknown shape {spec['shape']}")
    return spec["center"] + SHAPE_SCALE * raw


def sample_source_branch(rng, n, name):
    """Sample n points from branch name's initial (t=0) Gaussian.

    Args:
        rng: NumPy random Generator.
        n: Number of points.
        name: Branch key into SOURCE_CONFIGS.

    Returns:
        np.ndarray: Points, shape (n, 2).
    """
    cfg = SOURCE_CONFIGS[name]
    return rng.normal(cfg["center"], cfg["std"], size=(n, 2))


def main():
    """Generate the ablation dataset, write it to OUT_CSV, and save a scatter-plot preview.

    Args:
        None.

    Returns:
        None.
    """
    rng = np.random.default_rng(SEED)

    paths = {}
    for name, spec in BRANCHES.items():
        n_max = spec["n_max"]
        source_pts = sample_source_branch(rng, n_max, name)
        target_pts = sample_shape(rng, n_max, spec)
        paths[name] = (source_pts, target_pts)

    rows = []
    branch_data = {t: {} for t in range(N_STEPS)}

    for i in range(N_STEPS):
        alpha = ALPHAS[i]
        for name in BRANCHES:
            n_b = BRANCH_COUNTS[name][i]
            source_pts, target_pts = paths[name]
            source_pts = source_pts[:n_b]
            target_pts = target_pts[:n_b]

            bridge_std = BRIDGE_SIGMA * np.sqrt(alpha * (1 - alpha))
            noise = rng.normal(0.0, bridge_std, size=(n_b, 2))
            pos = (1 - alpha) * source_pts + alpha * target_pts + noise

            branch_data[i][name] = pos

            for x1, x2 in pos:
                rows.append((float(i), x1, x2))

    df = pd.DataFrame(rows, columns=["samples", "x1", "x2"])
    df.to_csv(OUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUT_CSV}")
    print(df.groupby("samples").size())

    # --- Scatter Plot Visualization ---
    colors = {
        "up": "#1f77b4",     # Muted blue
        "left": "#2ca02c",   # Soft green
        "right": "#d62728",  # Soft red
    }
    labels = {
        "up": "Up (Swiss Roll)",
        "left": "Left (Rings)",
        "right": "Right (Pinwheel)"
    }

    fig, axes = plt.subplots(1, 5, figsize=(18, 4.2), sharex=True, sharey=True, dpi=300)
    fig.patch.set_facecolor('#f8f9fa')

    for t in range(N_STEPS):
        ax = axes[t]
        ax.set_facecolor('#ffffff')
        total_pts = 0
        for name in ["up", "left", "right"]:
            pts = branch_data[t][name]
            total_pts += len(pts)
            ax.scatter(
                pts[:, 0], pts[:, 1],
                c=colors[name],
                s=12,
                alpha=0.75,
                edgecolors='none',
                label=labels[name] if t == 0 else ""
            )

        ax.set_title(f"t = {t}\n(N = {total_pts})", fontsize=12, fontweight='bold', pad=10)
        ax.set_xlim(-2.0, 2.0)
        ax.set_ylim(-1.5, 2.2)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(True, linestyle='--', alpha=0.4, color='#cccccc')
        ax.set_xlabel("$x_1$", fontsize=10)
        if t == 0:
            ax.set_ylabel("$x_2$", fontsize=10)

    # Figure legend
    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, legend_labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 1.08),
        ncol=3,
        frameon=True,
        facecolor='#ffffff',
        edgecolor='#dddddd',
        fontsize=11
    )

    plt.suptitle("Synthetic 2D Ablation Dataset Time Evolution (Distinct Initial Gaussians)", y=1.15, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(OUT_PNG, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"Saved visualization to {OUT_PNG}")


if __name__ == "__main__":
    main()
