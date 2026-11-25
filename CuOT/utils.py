"""Shared helpers for growth-field bookkeeping and result logging."""

import os
import json
import torch


def compute_delta_s(model, x_pred, device):
    """Compute the per-edge action difference delta_s along a predicted trajectory.

    Args:
        model: NeuralMap providing the vector_field used to evaluate growth.
        x_pred: Predicted trajectory, shape (n_nodes, n_samples, dim).
        device: Device to compute on.

    Returns:
        torch.Tensor: delta_s, shape (n_nodes - 1, n_samples).
    """
    n_nodes, n_samples, _ = x_pred.shape
    delta_s = torch.zeros(n_nodes - 1, n_samples).to(device)

    log_g_all = compute_g_grid(model, x_pred, device)  # (samples, node, time)

    s0_pred = torch.ones(n_samples, n_nodes).to(device)

    for j in range(n_nodes - 2, -1, -1):
        # delta_s = s(x_{j+1}, t_{j+1}) - s(x_{j}, t_{j})
        log_g_vec_p1 = log_g_all[:, j+1, :j+1]
        log_g_vec = log_g_all[:, j, :j]
        s0_p1 = s0_pred[:, j+1]
        s0 = s0_pred[:, j]
        a_p1 = torch.exp(log_g_vec_p1 / n_nodes)
        a = torch.exp(log_g_vec / n_nodes)
        s_p1, s = scan_simple(a_p1, a, s0_p1, s0)
        delta_s[j:j+1, :] = s_p1 - s

    return delta_s


def compute_g_grid(model, x_pred, device):
    """Evaluate the model's log-growth field over every (node, time) pair of a trajectory.

    Args:
        model: NeuralMap providing the vector_field to evaluate.
        x_pred: Predicted trajectory, shape (n_nodes, n_samples, dim).
        device: Device to compute on.

    Returns:
        torch.Tensor: log_g_all, shape (n_samples, n_nodes, n_nodes).
    """
    n_nodes, n_samples, dim = x_pred.shape
    t = torch.linspace(0, 1, n_nodes, device=device)
    x_all = x_pred.permute(1, 0, 2).unsqueeze(2)   # (n_samples, n_nodes, 1, dim)
    x_all = x_all.expand(-1, -1, n_nodes, -1)      # (n_samples, n_nodes, n_nodes, dim)
    t_all = t.view(1, 1, n_nodes).expand(n_samples, n_nodes, n_nodes)
    x_flat = x_all.reshape(n_samples * n_nodes, n_nodes, dim)
    t_flat = t_all.reshape(n_samples * n_nodes, n_nodes)
    _, log_g_all = model.vector_field(x_flat, t_flat)
    log_g_all = log_g_all.view(n_samples, n_nodes, n_nodes)  # (samples, node, time)
    return log_g_all


def scan_loop(a, a_m1, s, s_m1, b_prev, n_nodes):
    """Sequentially accumulate s and s_m1 across a[:, :j], tracking squared deltas.

    Args:
        a, a_m1: Per-step multiplicative factors, shape (n_samples, j).
        s, s_m1: Initial state values, shape (n_samples,).
        b_prev: Additive term for s's update, shape (n_samples, j).
        n_nodes: Total number of trajectory nodes, used to scale updates.

    Returns:
        Tuple (s, s_m1, delta_sq_next): final states and squared deltas, the
        latter shape (n_samples, j).
    """
    n_samples, j = a.shape
    delta_sq_next = torch.zeros(n_samples, j, device=a.device)

    for i in range(j):
        diff = s - s_m1
        delta_sq_next[:, i] = diff * diff
        s = s * a[:, i] + 0.5 * b_prev[:, i] / n_nodes
        s_m1 = s_m1 * a_m1[:, i] + 0.5 * delta_sq_next[:, i] / n_nodes

    return s, s_m1, delta_sq_next


def scan_simple(a, a_m1, s, s_m1):
    """Closed-form equivalent of scan_loop's final state, via cumulative products.

    Args:
        a, a_m1: Per-step multiplicative factors, shape (n_samples, j).
        s, s_m1: Initial state values, shape (n_samples,).

    Returns:
        Tuple (s_final, s_m1_final), each shape (n_samples,).
    """
    s_final = s * torch.prod(a, dim=1)
    s_m1_final = s_m1 * torch.prod(a_m1, dim=1)
    return s_final, s_m1_final


def logging(pos_w2, pos_w1, mass, seed, save_path):
    """Print per-marginal W1/W2 distance and mass metrics, and optionally save them as JSON.

    Args:
        pos_w2: Per-marginal Sinkhorn W2 position errors.
        pos_w1: Per-marginal Sinkhorn W1 position errors.
        mass: Per-marginal squared mass-ratio errors.
        seed: Random seed used for the run, saved alongside the results.
        save_path: If truthy, JSON output path (parent dirs are created).

    Returns:
        None.
    """
    results = []
    for j in range(len(pos_w2)):
        results.append({
            "marginal": j,
            "pos_w2": float(pos_w2[j]),
            "pos_w1": float(pos_w1[j]),
            "mass": float(mass[j]),
        })

    for r in results:
        print(
            f"marginal n {r['marginal']}: "
            f"pos_w2 {r['pos_w2']}, pos_w1 {r['pos_w1']}, mass {r['mass']}."
        )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print("Saving results to:", os.path.abspath(save_path))
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({"seed": seed, "results": results}, f, indent=2)
