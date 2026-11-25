"""Quantitative evaluation of a fitted trajectory against reference marginals."""

import torch
from geomloss import SamplesLoss


def quantitative_test(
    dataset,
    results,
    p,
    start_time: float = 0.0,
    end_time: float = 1.0
):
    """Compute per-marginal Sinkhorn position error and squared mass error.

    Args:
        dataset: Dataset providing reference marginals via get_all_particles_batch().
        results: Predicted trajectory, shape (n_steps, n_samples, dim + 1),
            with mass in the last channel.
        p: Order of the Sinkhorn/Wasserstein loss (1 or 2).
        start_time: Time of the first reference marginal.
        end_time: Time of the last reference marginal.

    Returns:
        Tuple of (evaluation_pos, evaluation_mass), one entry per reference marginal.
    """
    loss = SamplesLoss("sinkhorn", p=p, blur=0.05, scaling=0.9)

    results_tensor = results.detach()
    x_pred = results_tensor[..., :-1]
    m_pred = results_tensor[..., -1]
    _, _, dim = x_pred.shape
    m_pred_sum = torch.sum(m_pred, dim=-1) / torch.sum(m_pred[0], dim=-1)
    print("m_pred_sum: ", m_pred_sum)
    t_pred = torch.linspace(start_time, end_time, x_pred.shape[0])

    all_data = dataset.get_all_particles_batch()
    x_ref = list(all_data.values())

    if dataset.relative_mass is None:
        m_ref = [data.shape[0] / x_ref[0].shape[0] for data in x_ref]
    else:
        m_ref = dataset.relative_mass

    t_ref = torch.linspace(start_time, end_time, len(x_ref))

    # Debiased Sinkhorn also builds an (N, N) reference-vs-reference cost matrix. That's
    # independent of dim, but for datasets with tens of thousands of reference points (e.g.
    # FFHQ's ~57k adult samples) it alone is several GB, so cap how many reference points
    # feed the loss.
    ref_batch_max = 8000

    evaluation_pos = []
    evaluation_mass = []
    for idx, t in enumerate(t_ref):
        closest_idx = torch.argmin(torch.abs(t - t_pred))

        x_ref_idx = x_ref[idx][:ref_batch_max, :dim]
        ref_weights = (torch.ones(x_ref_idx.shape[0]) / x_ref_idx.shape[0]).to(x_pred.device)
        pred_weights = m_pred[closest_idx] / torch.sum(m_pred[closest_idx])

        measure_pos = loss(pred_weights, x_pred[closest_idx], ref_weights, x_ref_idx) ** (1 / p)
        evaluation_pos.append(measure_pos)

        measure_mass = (m_pred_sum[closest_idx] - m_ref[idx]).pow(2)
        evaluation_mass.append(measure_mass)

    return evaluation_pos, evaluation_mass
