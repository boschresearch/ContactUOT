"""Plotting utilities for trajectories, per-timestep marginals, growth fields, and
ablation-specific diagnostics (mode leakage, drift/growth vs. optimality curves)."""

from typing import Optional, List
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns
from datasets.generate_ablation import BRANCHES, SOURCE_CONFIGS


def plot_combined_data(data_points, generated_points, output_file="plot.png", highlight_times=None):
    """Scatter-plot reference data (X markers) against generated trajectories
    (mass-weighted, outlined dots) at the given highlight_times.

    Args:
        data_points: Reference points, shape (T_data, batch, 2).
        generated_points: Generated (position, mass) trajectory, shape
            (T_gen, batch, 3) (2D position + mass in the last channel).
        output_file: Output image path.
        highlight_times: Times (within [0, 1]) to draw from generated_points;
            defaults to [0, 0.5, 1.0].

    Returns:
        None.
    """
    if highlight_times is None:
        highlight_times = [0., 0.5, 1.]
    cmap = plt.cm.viridis
    sns.set_theme(style="white")

    plt.rcParams.update({
        'axes.prop_cycle': plt.cycler(color=cmap(np.linspace(0, 1, 10))),
        'axes.axisbelow': False,
        'axes.edgecolor': 'lightgrey',
        'axes.facecolor': 'None',
        'axes.grid': False,
        'axes.labelcolor': 'dimgrey',
        'axes.spines.right': False,
        'axes.spines.top': False,
        'figure.facecolor': 'white',
        'lines.solid_capstyle': 'round',
        'patch.edgecolor': 'w',
        'patch.force_edgecolor': True,
        'text.color': 'dimgrey',
        'xtick.bottom': False,
        'xtick.color': 'dimgrey',
        'xtick.direction': 'out',
        'xtick.top': False,
        'ytick.color': 'dimgrey',
        'ytick.direction': 'out',
        'ytick.left': False,
        'ytick.right': False,
        'font.size': 12,
        'axes.titlesize': 10,
        'axes.labelsize': 12
    })

    _, ax = plt.subplots(figsize=(12, 8), dpi=300)

    T_data = data_points.shape[0]
    T_gen, batch_size, dim = generated_points.shape
    T_high = len(highlight_times)

    colors_data = cmap(np.linspace(0, 1, T_data))
    colors_gen = cmap(np.linspace(0, 1, T_high))

    # marker styling (same trick as original code)
    size = 50
    size_x = 300
    outline_width = (0.3, 0.1)
    bg_width, gap_width = outline_width
    point = np.sqrt(size)

    gap_size = (point + (point * gap_width) * 2) ** 2
    bg_size = (np.sqrt(gap_size) + (point * bg_width) * 2) ** 2

    # --------------------
    # Ground truth points
    # --------------------
    for i in range(T_data):
        x = data_points[i, :, 0]
        y = data_points[i, :, 1]

        ax.scatter(
            x, y,
            marker='X',
            s=size_x,
            color=colors_data[i],
            alpha=0.2,
            linewidths=0
        )

    # --------------------
    # Generated points
    # --------------------

    # Highlight specific time points if requested
    generated = np.zeros([T_high, batch_size, dim])
    t_samples_np = np.linspace(highlight_times[0], highlight_times[-1], T_gen)
    for idx_h, t_high in enumerate(highlight_times):
        closest_idx = np.argmin(np.abs(t_samples_np - t_high))
        generated[idx_h] = generated_points[closest_idx]

    mass = generated[:, :, 2]
    mass_max_value = np.percentile(mass, 99)
    norm_mass = mass / (mass_max_value + 1e-8)
    norm_mass[norm_mass > 1.] = 1.
    minimum_value = 0.
    rescaled_minimum = 0.75
    mask = norm_mass >= minimum_value
    # rescale values >= 0.3 to [0.8, 1]
    norm_mass[mask] = rescaled_minimum + (norm_mass[mask] - minimum_value) / (1 - minimum_value) * (1 - rescaled_minimum)
    # set the rest to 0
    norm_mass[~mask] = 0

    for i in range(T_high):
        x = generated[i, :, 0]
        y = generated[i, :, 1]
        alpha = norm_mass[i]

        # outer black border
        ax.scatter(
            x, y,
            s=bg_size,
            c='black',
            alpha=alpha,
            linewidths=0
        )

        # white gap
        ax.scatter(
            x, y,
            s=gap_size,
            c='white',
            alpha=alpha,
            linewidths=0
        )

        # colored center
        ax.scatter(
            x, y,
            s=size,
            c=[colors_gen[i]],
            alpha=alpha,
            linewidths=0
        )

    # --------------------
    # Limits
    # --------------------
    min_x = min(data_points[:, :, 0].min(), generated_points[:, :, 0].min())
    max_x = max(data_points[:, :, 0].max(), generated_points[:, :, 0].max())
    min_y = min(data_points[:, :, 1].min(), generated_points[:, :, 1].min())
    max_y = max(data_points[:, :, 1].max(), generated_points[:, :, 1].max())
    span_x = max_x - min_x
    span_y = max_y - min_y
    ax.set_xlim(min_x - span_x*0.1, max_x + span_x*0.1)
    ax.set_ylim(min_y - span_y*0.1, max_y + span_y*0.1)

    # --------------------
    # Clean axes
    # --------------------
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_frame_on(False)
    ax.patch.set_alpha(0)

    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def visualizeresult(
    dataset,
    results: torch.Tensor,
    save_path: str,
    dim_reduction,
    highlight_times: Optional[List[float]] = None,
    visualization_batch: int = 200
    ):
    """
    Visualize 2D trajectories with mass-based coloring.

    Args:
        dataset: object with method sample_particles_batch(batch_size)
        results: torch.Tensor of shape [T, B, D+1] (last dim = mass)
        save_path: str, path to save the figure
        highlight_times: list of time points to highlight
        visualization_batch: number of particles to sample for background

    Returns:
        None.
    """
    # Convert results to CPU NumPy arrays once
    results_np = results.detach().cpu().numpy()
    x_traj = results_np[..., :-1]  # [T, B, D]
    m_traj = results_np[..., -1:]   # [T, B]

    num_timesteps, batch_size, dim = x_traj.shape

    if dim > 2:
        x_traj_flat = x_traj.reshape(-1, dim)

        all_data = dataset.get_all_particles_batch()
        # Move to CPU before concatenating -- these are only ever used as a
        # numpy array below, so there's no reason to hold the full dataset
        # concatenated on GPU (can be multiple GB at high dim).
        sample_list = [all_data[key][..., :dim].cpu().detach().numpy() for key in all_data]
        sample_np = np.concatenate(sample_list, axis=0)

        if dim_reduction == "umap":
            umap_reducer = umap.UMAP(n_components=2)
            umap_reducer.fit(sample_np)
            x_traj_flat_2d = umap_reducer.transform(x_traj_flat)
            reducer_obj = umap_reducer
        elif dim_reduction == "pca":
            pca_reducer = PCA(n_components=2)
            pca_reducer.fit(sample_np)
            x_traj_flat_2d = pca_reducer.fit_transform(x_traj_flat)
            reducer_obj = pca_reducer
        elif dim_reduction == "raw":
            class Reducer:
                """No-op reducer: just keeps the first two dimensions."""
                def transform(self, x):
                    """Return x's first two dimensions."""
                    return x[..., :2]
            reducer_obj = Reducer()
            x_traj_flat_2d = reducer_obj.transform(x_traj_flat)
        else:
            raise ValueError

        x_traj_2d = x_traj_flat_2d.reshape(num_timesteps, batch_size, 2)
    else:
        x_traj_2d = x_traj
        reducer_obj = None

    results_2d = np.concatenate([x_traj_2d[:, :visualization_batch], m_traj[:, :visualization_batch]], axis=-1)

    # Sample background particles from dataset
    dictionary = dataset.sample_particles_batch(visualization_batch)
    sampled_data = list(dictionary.values())
    data_points = np.zeros([len(sampled_data), visualization_batch, 2])
    for i, particle in enumerate(sampled_data):
        particle_np = particle.detach().cpu().numpy()
        if dim > 2 and reducer_obj is not None:
            particle_2d = reducer_obj.transform(particle_np[..., :dim])
        else:
            particle_2d = particle_np
        data_points[i] = particle_2d

    plot_combined_data(data_points, results_2d, output_file=save_path, highlight_times=highlight_times)


def visualize_per_timestep(
    dataset,
    results: torch.Tensor,
    save_path: str,
    dim_reduction,
    highlight_times: Optional[List[float]] = None,
    visualization_batch: int = 200
    ):
    """
    Small-multiples view: one panel per observed time step, each showing the
    reference distribution (grey) against the model's predicted distribution
    (colored) at the matching integration node. Complements visualizeresult(),
    which overlays every time step in a single panel and can be hard to read
    for intricate or multi-modal targets.

    Args:
        dataset: Dataset providing get_all_particles_batch()/get_certain_time_data().
        results: Predicted trajectory, shape (T, B, D+1) (last dim = mass).
        save_path: Output image path.
        dim_reduction: "umap", "pca", or "raw" (keep first 2 dims), used when D > 2.
        highlight_times: Normalized (0..1) times to plot; defaults to dataset's
            own time_steps / max_time.
        visualization_batch: Number of predicted particles to show per panel.

    Returns:
        None.
    """
    results_np = results.detach().cpu().numpy()
    x_traj = results_np[..., :-1]  # [T, B, D]

    num_timesteps, batch_size, dim = x_traj.shape

    if highlight_times is None:
        highlight_times = list(dataset.time_steps / dataset.max_time)

    if dim > 2:
        x_traj_flat = x_traj.reshape(-1, dim)

        all_data = dataset.get_all_particles_batch()
        # Move to CPU before concatenating -- these are only ever used as a
        # numpy array below, so there's no reason to hold the full dataset
        # concatenated on GPU (can be multiple GB at high dim).
        sample_list = [all_data[key][..., :dim].cpu().detach().numpy() for key in all_data]
        sample_np = np.concatenate(sample_list, axis=0)

        if dim_reduction == "umap":
            umap_reducer = umap.UMAP(n_components=2)
            umap_reducer.fit(sample_np)
            x_traj_flat_2d = umap_reducer.transform(x_traj_flat)
            reducer_obj = umap_reducer
        elif dim_reduction == "pca":
            pca_reducer = PCA(n_components=2)
            pca_reducer.fit(sample_np)
            x_traj_flat_2d = pca_reducer.fit_transform(x_traj_flat)
            reducer_obj = pca_reducer
        elif dim_reduction == "raw":
            class Reducer:
                """No-op reducer: just keeps the first two dimensions."""
                def transform(self, x):
                    """Return x's first two dimensions."""
                    return x[..., :2]
            reducer_obj = Reducer()
            x_traj_flat_2d = reducer_obj.transform(x_traj_flat)
        else:
            raise ValueError

        x_traj_2d = x_traj_flat_2d.reshape(num_timesteps, batch_size, 2)
    else:
        x_traj_2d = x_traj
        reducer_obj = None

    n_panels = len(highlight_times)
    t_nodes = np.linspace(0.0, 1.0, num_timesteps)
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0, 1, n_panels))

    fig, axes = plt.subplots(1, n_panels, figsize=(4.2 * n_panels, 4.4), sharex=True, sharey=True)
    if n_panels == 1:
        axes = [axes]

    for idx, (ax, t_frac, real_t) in enumerate(zip(axes, highlight_times, dataset.time_steps)):
        node = int(np.argmin(np.abs(t_nodes - t_frac)))

        ref = dataset.get_certain_time_data(real_t)
        if hasattr(ref, "detach"):
            ref = ref.detach().cpu().numpy()
        if dim > 2 and reducer_obj is not None:
            ref_2d = reducer_obj.transform(ref[..., :dim])
        else:
            ref_2d = ref

        pred_2d = x_traj_2d[node, :visualization_batch]

        ax.scatter(ref_2d[:, 0], ref_2d[:, 1], s=8, color="lightgray", alpha=0.6, linewidths=0, label="reference")
        ax.scatter(pred_2d[:, 0], pred_2d[:, 1], s=8, color=colors[idx], alpha=0.7, linewidths=0, label="predicted")
        ax.set_title(f"t={real_t:g}")
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    axes[0].legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


_CLUSTER_PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# Ablation-specific: the dataset's own shape name -> the label to display.
_SHAPE_DISPLAY_NAME = {"swiss_roll": "spiral", "rings": "rings", "pinwheel": "pinwheel"}


def _nearest_cluster(points, centers):
    """Assign each point to the index of its nearest center.

    Args:
        points: Points, shape (N, d).
        centers: Cluster centers, shape (K, d).

    Returns:
        np.ndarray: Nearest-center index per point, shape (N,).
    """
    dists = np.linalg.norm(points[:, None, :] - centers[None, :, :], axis=-1)
    return np.argmin(dists, axis=-1)


def compute_ablation_target_labels(x_final):
    """
    Cluster ablation-experiment samples by nearest known target shape center
    -- same nearest-center logic (and same reasoning for not using
    unsupervised clustering) as the target side of
    visualize_ablation_mode_leakage.

    Args:
        x_final: (batch, 2) array of final (t=1) positions.

    Returns:
        labels: (batch,) int array, values in range(len(BRANCHES))
        display_names: list of str, index-aligned with the label ids
    """
    branch_names = list(BRANCHES.keys())
    display_names = [_SHAPE_DISPLAY_NAME[BRANCHES[n]["shape"]] for n in branch_names]
    target_centers = np.stack([BRANCHES[n]["center"] for n in branch_names], axis=0)
    labels = _nearest_cluster(x_final, target_centers)
    return labels, display_names


def visualize_ablation_mode_leakage(
    results: torch.Tensor,
    save_path: str,
    tau: float = 1.0,
    ):
    """
    Mode-leakage diagnostics, specific to the "ablation" experiment's three
    source Gaussians -> {spiral, rings, pinwheel} flow.

    Cluster membership is NOT discovered via unsupervised clustering (k-means
    would need to guess boundaries, and could misassign points near a
    spiral's tail or a pinwheel's arm tips) -- it uses the dataset's own known
    geometry instead: "source" cluster = nearest of the three known initial
    Gaussian centers (datasets.generate_ablation.SOURCE_CONFIGS) to the
    sample's t=0 position; "target" cluster = nearest of the three known
    final shape centers (datasets.generate_ablation.BRANCHES) to the model's
    predicted final position. Since both classifications are deterministic
    nearest-center lookups against fixed, known points, the leakage metric
    reported here is not polluted by clustering-algorithm mistakes -- any
    off-diagonal mass in the confusion matrix reflects the model's own
    transport plan, not an artifact of how clusters were discovered.

    Saves a single PDF with three panels:
      1. average velocity per (target-)cluster, over time
      2. one pie chart per target cluster (spiral / rings / pinwheel), sliced
         by which source cluster its members came from (the leakage picture)
      3. average mass per (target-)cluster, over time
    "Belonging" to a cluster is always defined by where a sample lands (its
    target-cluster membership).

    Args:
        results: torch.Tensor [T, B, D+1] (last dim = mass), as returned by
            NeuralMap.forward, for the ablation dataset (D=2).
        save_path: output path (use a .pdf extension).
        tau: integration step size (t1 / n_steps) used to convert node-to-node
            displacement into a physical velocity. Defaults to 1.0 (i.e. plain
            per-node displacement) if not provided.

    Returns:
        dict with the source x target confusion matrix, the overall leakage
        rate, and the cluster labels (in confusion-matrix order).
    """
    results_np = results.detach().cpu().numpy()
    x_traj = results_np[..., :-1]   # [T, B, 2]
    m_traj = results_np[..., -1]    # [T, B]
    n_nodes = x_traj.shape[0]

    branch_names = list(BRANCHES.keys())  # e.g. ["up", "left", "right"]
    n_clusters = len(branch_names)
    labels_display = [_SHAPE_DISPLAY_NAME[BRANCHES[n]["shape"]] for n in branch_names]

    source_centers = np.stack([SOURCE_CONFIGS[n]["center"] for n in branch_names], axis=0)
    target_centers = np.stack([BRANCHES[n]["center"] for n in branch_names], axis=0)

    source_labels = _nearest_cluster(x_traj[0], source_centers)
    target_labels = _nearest_cluster(x_traj[-1], target_centers)

    confusion = np.zeros((n_clusters, n_clusters), dtype=int)
    for s in range(n_clusters):
        for t in range(n_clusters):
            confusion[s, t] = int(np.sum((source_labels == s) & (target_labels == t)))
    total = confusion.sum()
    leakage_rate = 1.0 - np.trace(confusion) / total if total > 0 else 0.0

    colors = [_CLUSTER_PALETTE[c % len(_CLUSTER_PALETTE)] for c in range(n_clusters)]

    vel = np.linalg.norm(np.diff(x_traj, axis=0), axis=-1) / tau   # [T-1, B]
    t_vel = np.linspace(0.0, 1.0, n_nodes - 1)
    t_mass = np.linspace(0.0, 1.0, n_nodes)

    fig = plt.figure(figsize=(4.4 * (2 + n_clusters), 4.6), dpi=200)
    gs = fig.add_gridspec(1, 2 + n_clusters, width_ratios=[1.3] + [1.0] * n_clusters + [1.3])

    ax_vel = fig.add_subplot(gs[0, 0])
    for c, name in enumerate(labels_display):
        sel = target_labels == c
        curve = vel[:, sel].mean(axis=-1) if sel.any() else np.zeros(vel.shape[0])
        ax_vel.plot(t_vel, curve, color=colors[c], linewidth=2.0, label=name)
    ax_vel.set_title("Average velocity per cluster")
    ax_vel.set_xlabel("Normalized time")
    ax_vel.set_ylabel("|velocity|")
    ax_vel.legend(fontsize=8, frameon=False)

    for c, name in enumerate(labels_display):
        ax_pie = fig.add_subplot(gs[0, 1 + c])
        slice_counts = confusion[:, c]
        if slice_counts.sum() == 0:
            ax_pie.text(0.5, 0.5, "no samples", ha="center", va="center", fontsize=9)
            ax_pie.axis("off")
            continue
        ax_pie.pie(slice_counts, colors=colors, startangle=90,
                   autopct=lambda p: f"{p:.0f}%" if p >= 3 else "",
                   wedgeprops=dict(linewidth=1, edgecolor="white"))
        ax_pie.set_title(f"target: {name}\n(by source origin)", fontsize=9.5)

    ax_mass = fig.add_subplot(gs[0, 1 + n_clusters])
    for c, name in enumerate(labels_display):
        sel = target_labels == c
        curve = m_traj[:, sel].mean(axis=-1) if sel.any() else np.zeros(m_traj.shape[0])
        ax_mass.plot(t_mass, curve, color=colors[c], linewidth=2.0, label=name)
    ax_mass.set_title("Average mass per cluster")
    ax_mass.set_xlabel("Normalized time")
    ax_mass.set_ylabel("mean mass")
    ax_mass.legend(fontsize=8, frameon=False)

    fig.suptitle(f"Mode-leakage diagnostics  (overall leakage rate: {leakage_rate:.1%})", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(save_path)
    plt.close(fig)

    return dict(confusion=confusion.tolist(), leakage_rate=float(leakage_rate), labels=labels_display)


def visualizedatagrowth(
    dataset,
    model,
    save_path: str,
    dim_reduction,
    dim,
    n_steps,
    t1
    ):
    """Scatter-plot every dataset sample colored by the model's predicted growth rate.

    Args:
        dataset: Dataset providing get_all_particles_batch().
        model: Trained NeuralMap providing vector_field.
        save_path: Output image path.
        dim_reduction: "umap", "pca", or "raw" (keep first 2 dims), used when dim > 2.
        dim: Spatial dimension of the data (excluding any conditioning channels).
        n_steps: Number of integration steps used at inference.
        t1: Integration end time, used to convert log-growth-rate to log-growth.

    Returns:
        None.
    """
    device = next(model.parameters()).device
    all_data = dataset.get_all_particles_batch()
    sample_list = []
    for key in all_data:
        data_np = all_data[key].cpu().detach().numpy()
        time_col = np.full((data_np.shape[0], 1), key, dtype=data_np.dtype)
        sample_list.append(np.concatenate([data_np, time_col], axis=-1))
    sample_np = np.concatenate(sample_list, axis=0)
    sample_np[:, -1:] = sample_np[:, -1:] / key
    # CPU-resident; only the current growth_batch_size-sized slice is moved
    # to the GPU below, so the full (N, dim+1) tensor never has to fit on
    # the device at once (can be multiple GB at high dim).
    sample_all = torch.from_numpy(sample_np)

    # Batched, no-grad evaluation: the Fourier expansion inflates the input
    # width by ~(1 + 2*n_frequencies)x, which OOMs on the whole dataset at
    # once for high-dim latents (e.g. the 9216-d pSp W+ space). Chunking is
    # exact here since each sample's growth is computed independently.
    growth_batch_size = max(1, min(4096, 1_000_000 // dim))
    log_g_chunks = []
    with torch.no_grad():
        for start in range(0, sample_all.shape[0], growth_batch_size):
            end = start + growth_batch_size
            batch = sample_all[start:end].to(device)
            _, log_g_chunk = model.vector_field(
                batch[:, :dim].unsqueeze(1), batch[:, -1:]
            )
            log_g_chunks.append(log_g_chunk)
    log_g_all = torch.cat(log_g_chunks, dim=0)
    growth = torch.exp(log_g_all * t1/n_steps).squeeze().cpu().detach().numpy() - 1
    vmax_value = np.percentile(growth, 99)

    if dim > 2:
        if dim_reduction == "umap":
            umap_reducer = umap.UMAP(n_components=2)
            umap_reducer.fit(sample_np[:, :dim])
            x_traj_2d = umap_reducer.transform(sample_np[:, :dim])
        elif dim_reduction == "pca":
            pca_reducer = PCA(n_components=2)
            pca_reducer.fit(sample_np[:, :dim])
            x_traj_2d = pca_reducer.fit_transform(sample_np[:, :dim])
        elif dim_reduction == "raw":
            class Reducer:
                """No-op reducer: just keeps the first two dimensions."""
                def transform(self, x):
                    """Return x's first two dimensions."""
                    return x[..., :2]
            reducer_obj = Reducer()
            x_traj_2d = reducer_obj.transform(sample_np[:, :dim])
        else:
            raise ValueError

    else:
        x_traj_2d = sample_np[:, :-1]

    xs = x_traj_2d[:, 0]
    ys = x_traj_2d[:, 1]

    fig = plt.figure(figsize=(8, 6))
    sc = plt.scatter(xs, ys, c=growth, cmap=sns.color_palette("icefire", as_cmap=True),
                 marker='o', alpha=0.7, vmax=vmax_value)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    cbar = plt.colorbar(sc)
    cbar.ax.tick_params(labelsize=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def visualize_drift_and_growth(xm_pred, v_opt, z_opt, growth, save_path,
                                cluster_labels=None, cluster_names=None):
    """
    Compare the model's optimality diagnostics (v_opt, z_opt from
    NeuralMap.compute_v_S_z, computed at eval / "optimality" -- not tracked
    during training) against what the generated trajectory actually does:
    the real step-to-step displacement of the generated samples, and the
    growth rate implied by the generated mass trajectory.

    v_opt scales the generated drift in the training loss (jacobi_distance =
    v_opt**2 * distance), and z_opt is fit to track growth via z_opt[1:] ~
    growth -- this plots both pairs so it's visible how closely the
    relationship holds on real generated trajectories.

    Args:
        xm_pred: (n_steps+1, batch, dim+1), model output with
            return_details=True (positions + mass concatenated).
        v_opt:   (n_steps, batch), from compute_v_S_z.
        z_opt:   (n_steps+1, batch), from compute_v_S_z.
        growth:  (n_steps, batch), actual growth rate backed out of the mass
            trajectory, also returned by compute_v_S_z.
        save_path: output path (no extension needed, matches other plots).
        cluster_labels: optional (batch,) int array grouping samples (e.g.
            by target shape for the ablation experiment, via
            compute_ablation_target_labels). When given, each panel shows
            one actual/modeled pair per cluster -- color identifies the
            cluster, line style/marker identifies actual vs. modeled --
            instead of a single batch-wide average.
        cluster_names: list of str, index-aligned with cluster_labels' ids.
            Required when cluster_labels is given; used for the legend.

    Returns:
        None.
    """
    x_traj = xm_pred[..., :-1].detach().cpu()
    v_opt = v_opt.detach().cpu()**2
    z_opt = z_opt.detach().cpu()
    growth = growth.detach().cpu()

    # actual step size: squared displacement between consecutive generated
    # positions, edge-indexed like v_opt -- the same pairing used in
    # training.py's jacobi_distance = v_opt**2 * distance.
    generated_drift = (((x_traj[1:] - x_traj[:-1]) * 20) ** 2).sum(dim=-1)

    if cluster_labels is None:
        cluster_ids = [None]
    else:
        cluster_labels = np.asarray(cluster_labels)
        cluster_ids = sorted(set(cluster_labels.tolist()))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    for cluster_id in cluster_ids:
        if cluster_id is None:
            sel = slice(None)
            suffix = ""
        else:
            sel = torch.as_tensor(cluster_labels == cluster_id)
            suffix = f" ({cluster_names[cluster_id]})"
        color = _CLUSTER_PALETTE[(cluster_id or 0) % len(_CLUSTER_PALETTE)]

        drift_curve = generated_drift[:, sel].mean(dim=-1).numpy()
        v_opt_curve = v_opt[:, sel].mean(dim=-1).numpy()
        growth_curve = growth[:, sel].mean(dim=-1).numpy()
        z_curve = z_opt[1:, sel].mean(dim=-1).numpy()
        steps = np.arange(1, len(v_opt_curve) + 1)

        axes[0].plot(steps, drift_curve, color=color, linestyle="-", linewidth=2,
                     marker="o", markersize=5, label=f"generated drift{suffix}")
        axes[0].plot(steps, v_opt_curve, color=color, linestyle="--", linewidth=2,
                     marker="x", markersize=6, label=f"v_opt{suffix}")

        axes[1].plot(steps, growth_curve, color=color, linestyle="-", linewidth=2,
                     marker="o", markersize=5, label=f"growth{suffix}")
        axes[1].plot(steps, z_curve, color=color, linestyle="--", linewidth=2,
                     marker="x", markersize=6, label=f"z_opt{suffix}")

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Squared displacement")
    axes[0].set_title("Generated drift vs. v_opt")
    axes[0].legend(fontsize=8, frameon=False)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Growth rate")
    axes[1].set_title("Generated growth vs. z_opt")
    axes[1].legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def visualizedataphi(
    dataset,
    results,
    drift,
    indexes_marginals,
    save_path,
    dim_reduction,
    ):
    """Print per-segment position/drift error breakdowns and scatter-plot phi = drift^2.

    Args:
        dataset: Dataset providing get_all_particles_batch(), used when dim > 2.
        results: Predicted trajectory, shape (T, B, D+1) (last dim = mass).
        drift: Per-step drift magnitude, shape (T-1, B, 1) (or similar).
        indexes_marginals: Integration-node indices bounding each printed segment
            (0 and the final node are added automatically).
        save_path: Output image path.
        dim_reduction: "umap" or "pca", used when D > 2.

    Returns:
        None.
    """
    results_np = results.detach().cpu().numpy()
    drift_np = drift.detach().cpu().numpy()
    x_traj = results_np[..., :-1]  # [T, B, D]
    m_traj = results_np[..., -1]   # [T, B]

    num_timesteps, _, dim = x_traj.shape

    distance = (((x_traj[1:, :] - x_traj[:-1, :])**2).sum(axis=-1))

    for j in range(num_timesteps-1):
        print(f"distance_2[{j}]: {distance[j].mean()} +- {distance[j].std()}")

    spacing = []
    indexes_marginals = [0] + indexes_marginals + [num_timesteps-1]
    for j in range(len(indexes_marginals)-1):
        spacing.append([indexes_marginals[j], indexes_marginals[j+1]])

    w2_dist = np.sqrt(distance.mean(axis=-1))
    for j in range(len(spacing)):
        start_i = spacing[j][0]
        end_i = spacing[j][1]
        rel_dist = w2_dist[start_i:end_i].sum() / w2_dist.sum()
        print("rel_dist: ", rel_dist)

    growth_rate = (np.diff(m_traj, axis=0) /(m_traj[:-1])).squeeze() # (1 / num_timesteps)
    growth_rate = np.concatenate([growth_rate, growth_rate[-1:]], axis=0)
    growth_rate = np.concatenate([growth_rate, growth_rate[-1:]], axis=0)

    drift_np = np.concatenate([drift_np, drift_np[-1:]], axis=0)
    phi = (drift_np**2).squeeze()

    for j in range(num_timesteps):
        print(f"growth_rate[{j}]: {growth_rate[j].mean()} +- {growth_rate[j].std()}")
    for j in range(num_timesteps-1):
        print(f"drift[{j}]: {drift_np[j].mean()} +- {drift_np[j].std()}")

    for j in range(len(spacing)):
        start_i = spacing[j][0]
        end_i = spacing[j][1]
        rel_phi = phi[start_i:end_i].sum() / phi.sum()
        print("rel_phi: ", rel_phi)
    x_traj = x_traj.reshape(-1, dim)

    if dim > 2:
        all_data = dataset.get_all_particles_batch()
        sample_list = [all_data[key].cpu().detach().numpy() for key in all_data]
        sample_np = np.concatenate(sample_list, axis=0)

        if dim_reduction == "umap":
            umap_reducer = umap.UMAP(n_components=2)
            umap_reducer.fit(sample_np)
            x_traj_2d = umap_reducer.transform(x_traj)
        elif dim_reduction == "pca":
            pca_reducer = PCA(n_components=2)
            pca_reducer.fit(sample_np)
            x_traj_2d = pca_reducer.fit_transform(x_traj)
        else:
            raise ValueError

    else:
        x_traj_2d = x_traj

    xs = x_traj_2d[:, 0]
    ys = x_traj_2d[:, 1]

    fig = plt.figure(figsize=(8, 6))
    sc = plt.scatter(xs, ys, c=phi.reshape(-1, 1).squeeze(), cmap='viridis', s=30, alpha=0.3)
    plt.xticks([])
    plt.yticks([])
    plt.xlabel("")
    plt.ylabel("")
    plt.colorbar(sc)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close(fig)


def scatter_pred_vs_ref(x_pred, x_ref, save_path=None, title=None):
    """Scatter-plot predicted vs. reference points; x_pred/x_ref are (batch, dim) tensors.

    Args:
        x_pred: Predicted points, shape (batch, dim).
        x_ref: Reference points, shape (batch, dim).
        save_path: If given, save to this path; else show interactively.
        title: Optional plot title.

    Returns:
        None.
    """
    x_pred = x_pred.detach().cpu()
    x_ref = x_ref.detach().cpu()

    plt.figure(figsize=(5, 5))

    plt.scatter(
        x_ref[:, 0], x_ref[:, 1],
        s=30, alpha=0.6, label="Reference", marker="o"
    )
    plt.scatter(
        x_pred[:, 0], x_pred[:, 1],
        s=30, alpha=0.6, label="Predicted", marker="x"
    )

    plt.axis("equal")
    plt.legend()
    plt.grid(True)

    if title is not None:
        plt.title(title)

    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def visualize_g_tensor(tensor, save_path):
    """Heatmap the mean per-(node, time) growth factor, with row-product and
    row-difference sidebars showing the cumulative and marginal growth per node.

    Args:
        tensor: Log-growth-rate grid, shape (samples, n_nodes, n_nodes), as
            returned by utils.compute_g_grid.
        save_path: Output image path.

    Returns:
        None.
    """
    tensor = torch.exp(tensor / tensor.shape[-1])
    tensor = tensor.detach().cpu().numpy()
    mean_tensor = tensor.mean(axis=0)  # Shape (21, 21)

    # Compute row products up to the diagonal
    masked_matrix = np.tril(mean_tensor, k=-1) + np.triu(np.ones(mean_tensor.shape), k=0)
    row_products = np.prod(masked_matrix, axis=1)
    row_diffs = np.diff(row_products, prepend=row_products[0])

    # --- Visualization ---
    fig, ax = plt.subplots(figsize=(12, 9))
    im = ax.imshow(mean_tensor, cmap='viridis')

    # Create a divider to add the row product column and the colorbar
    divider = make_axes_locatable(ax)

    # 1. Row products sidebar
    prod_ax = divider.append_axes("right", size="8%", pad=0.5)
    prod_column = row_products.reshape(-1, 1)
    prod_ax.imshow(prod_column, cmap='magma', aspect='auto')

    # 2. Row product differences sidebar
    diff_ax = divider.append_axes("right", size="8%", pad=0.2)
    diff_column = row_diffs.reshape(-1, 1)
    diff_ax.imshow(diff_column, cmap='magma', aspect='auto')

    # 2. Add Colorbar on the far right
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax).set_label('Cell Magnitude')

    # --- Annotations ---
    rows, cols = mean_tensor.shape
    # Main Matrix Annotations
    for i in range(rows):
        for j in range(cols):
            val = mean_tensor[i, j]
            color = "white" if (val - mean_tensor.min()) < (mean_tensor.max() - mean_tensor.min()) * 0.75 else "black"
            ax.text(j, i, f"{val:.1f}", ha="center", va="center", color=color, fontsize=8)


    # Row Product Annotations
    for i in range(rows):
        val = row_products[i]
        color = "white" if (val - row_products.min()) < (row_products.max() - row_products.min()) * 0.75 else "black"
        prod_ax.text(0, i, f"{val:.1e}" if val > 1000 else f"{val:.2f}",
                     ha="center", va="center", color=color, fontsize=8, fontweight='bold')


    # Row Diffs Annotations
    for i in range(rows):
        val = row_diffs[i]
        color = "white" if (val - row_diffs.min()) < (row_diffs.max() - row_diffs.min()) * 0.75 else "black"
        diff_ax.text(0, i, f"{val:+.1e}" if abs(val) > 1000 else f"{val:+.3f}",
                    ha="center", va="center", color=color, fontsize=8)

    # --- Formatting ---
    ax.set_xticks(np.arange(cols))
    ax.set_yticks(np.arange(rows))
    ax.set_title('Mean Magnitude per Cell')
    ax.set_xlabel('Time Index')
    ax.set_ylabel('Node Index')

    prod_ax.set_xticks([])
    prod_ax.set_yticks([])
    prod_ax.set_title('Row Prod')

    diff_ax.set_xticks([])
    diff_ax.set_yticks([])
    diff_ax.set_title('Δ Prod')

    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
