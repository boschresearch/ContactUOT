"""
Training utilities for fitting neural transport maps between probability
distributions using Sinkhorn divergence.
"""

from abc import ABC, abstractmethod
from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


class BaseTrainer(ABC):
    """Shared training loop (optimizer, AMP, checkpointing, loss plots) for a NeuralMap."""

    def __init__(self, model: nn.Module, dataset):
        """Store the model and dataset the trainer will fit.

        Args:
            model: The NeuralMap being trained.
            dataset: Dataset providing per-timestep particle batches.
        """
        self.model = model
        self.dataset = dataset
        self._mass_list_cache = None

    @abstractmethod
    def get_loss(self, xm_refs, plot=False, horizon=0) -> torch.Tensor:
        """Compute the training loss for a batch of reference marginals. Implemented by subclasses.

        Args:
            xm_refs: Batch from sample_batch(), (data_list, mass_list).
            plot: If True, print per-marginal loss diagnostics.
            horizon: Current training horizon, used to weight intermediate marginals.

        Returns:
            Tuple (loss, energy_term, marginal_term, extra) of scalar tensors.
        """

    def get_optimizer(self, lr: float):
        """Build the Adam optimizer used for training.

        Args:
            lr: Learning rate.

        Returns:
            torch.optim.Adam: Optimizer over the model's parameters.
        """
        return torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=1e-4
        )

    def sample_batch(self, batch_size):
        """Sample a batch of reference particles and their per-marginal mass weights.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            Tuple (data_list, mass_list): per-timestep particle tensors (mass
            appended to the first) and their reference population sizes.
        """
        dictionary_batch = self.dataset.sample_particles_batch(batch_size)
        data_list = list(dictionary_batch.values())
        mass_added = torch.cat([data_list[0], torch.ones_like(data_list[0][:, :1])], dim=1)
        data_list[0] = mass_added

        if self.dataset.relative_mass is None:
            # Only .shape[0] (population counts) is needed here, not the
            # actual data -- get_particle_counts() gives that without
            # moving the full dataset to GPU (get_all_particles_batch()
            # does, which is multi-GB at high dim and was OOM-ing here,
            # before training even reaches its own batching safeguards).
            if hasattr(self.dataset, "get_particle_counts"):
                counts = self.dataset.get_particle_counts()
            else:
                counts = {k: v.shape[0] for k, v in self.dataset.get_all_particles_batch().items()}
            mass_list = list(counts.values())
        else:
            mass_list = self.dataset.relative_mass
        return data_list, mass_list

    def train(self, num_epochs, num_batches, batch_size,
              n_steps, device, lr=1e-3, save_path="best_model.pt",
              loss_plot_name=None, amp_enabled=False, mode=0,
              checkpoint_path=None, checkpoint_every=100, resume_from=None):
        """Train the model, checkpointing periodically and keeping the best test-loss state.

        Args:
            num_epochs: Number of training epochs.
            num_batches: Number of batches sampled per epoch.
            batch_size: Particles sampled per timestep, per batch.
            n_steps: Number of Euler integration steps in the model.
            device: Device to train on.
            lr: Learning rate.
            save_path: Path to save the final model state.
            loss_plot_name: If set, base path for the loss-curve plots.
            amp_enabled: Enable mixed-precision training (CUDA only).
            mode: If 0, save/restore the best test-loss state; else the last state.
            checkpoint_path: Path for periodic checkpoints (derived from save_path if None).
            checkpoint_every: Save a checkpoint every this many epochs.
            resume_from: Checkpoint path to resume training from, if any.

        Returns:
            torch.Tensor: The per-epoch test-loss history.
        """
        self.model.to(device)
        self.model.train()

        opt = self.get_optimizer(lr)

        use_amp = device.type == "cuda" and amp_enabled
        scaler = torch.amp.GradScaler(enabled=use_amp)
        ref_test_samples = self.sample_batch(batch_size)

        if checkpoint_path is None:
            root, ext = save_path.rsplit(".", 1) if "." in save_path else (save_path, "pt")
            checkpoint_path = f"{root}_checkpoint.{ext}"

        best_loss = float("inf")
        best_state = None
        train_hist, test_hist = [], []
        energy_hist, int_hist, ter_hist = [], [], []
        horizon_step = (n_steps * 2) / num_epochs
        start_epoch = 0

        if resume_from is not None:
            ckpt = torch.load(resume_from, map_location=device)
            self.model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optimizer_state"])
            if use_amp and ckpt.get("scaler_state") is not None:
                scaler.load_state_dict(ckpt["scaler_state"])
            start_epoch = ckpt["epoch"] + 1
            best_loss = ckpt["best_loss"]
            best_state = ckpt["best_state"]
            train_hist, test_hist = ckpt["train_hist"], ckpt["test_hist"]
            energy_hist, int_hist, ter_hist = ckpt["energy_hist"], ckpt["int_hist"], ckpt["ter_hist"]

        pbar = tqdm(range(start_epoch, num_epochs), desc="Epochs")
        for epoch in pbar:
            epoch_loss = 0.0
            horizon = horizon_step * epoch

            for _ in range(num_batches):
                ref_train_samples = self.sample_batch(batch_size)

                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    loss, _, _, _ = self.get_loss(ref_train_samples, plot=False, horizon=horizon)

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    opt.step()

                epoch_loss += loss.item()

            epoch_loss /= num_batches
            train_hist.append(epoch_loss)

            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=use_amp):
                test_loss, energy, int_mar, ter_mar = (x.item() for x in self.get_loss(ref_test_samples, plot=False, horizon=horizon))

            test_hist.append(test_loss)

            energy_hist.append(energy)
            int_hist.append(int_mar)
            ter_hist.append(ter_mar)

            lr_now = opt.param_groups[0]["lr"]
            pbar.set_description(
                f"Epoch {epoch}, test_loss={test_loss:.4f}, train_loss={epoch_loss:.4f}, lr={lr_now:.2e}"
            )

            if test_loss < best_loss:
                best_loss = test_loss
                best_state = self.model.state_dict()

            if (epoch + 1) % checkpoint_every == 0 or epoch == num_epochs - 1:
                torch.save({
                    "epoch": epoch,
                    "model_state": self.model.state_dict(),
                    "optimizer_state": opt.state_dict(),
                    "scaler_state": scaler.state_dict() if use_amp else None,
                    "best_loss": best_loss,
                    "best_state": best_state,
                    "train_hist": train_hist,
                    "test_hist": test_hist,
                    "energy_hist": energy_hist,
                    "int_hist": int_hist,
                    "ter_hist": ter_hist,
                }, checkpoint_path)

            if lr_now < 1e-8:
                break

        if mode == 0 and best_state is not None:
            self.model.load_state_dict(best_state)
            torch.save(best_state, save_path)
        else:
            torch.save(self.model.state_dict(), save_path)

        if loss_plot_name:
            self._plot_losses(train_hist, test_hist, loss_plot_name)
            loss_plot_name_components = loss_plot_name + "_components"
            self._plot_losses_components(energy_hist, int_hist, ter_hist, loss_plot_name_components)

        return torch.tensor(test_hist)

    @staticmethod
    def _plot_losses(train, test, path):
        """Save a log-scale train-vs-test loss curve to path.

        Args:
            train: Per-epoch train-loss history.
            test: Per-epoch test-loss history.
            path: Output image path.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(train, label="Train")
        ax.plot(test, label="Test")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.savefig(path, dpi=300)
        plt.close(fig)

    @staticmethod
    def _plot_losses_components(energy, int_mar, term_mar, path):
        """Save a log-scale plot of the energy/intermediate/terminal loss components to path.

        Args:
            energy: Per-epoch energy-term history.
            int_mar: Per-epoch intermediate-marginal-loss history.
            term_mar: Per-epoch terminal-marginal-loss history.
            path: Output image path.
        """
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(energy, label="energy")
        ax.plot(int_mar, label="intermediate")
        ax.plot(term_mar, label="terminal")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.savefig(path, dpi=300)
        plt.close(fig)


class FitGeodesicProbability(BaseTrainer):
    """Trainer for fitting a transport map T_theta to a geodesic between distributions."""

    def __init__(self, dataset, loss_fn, loss_fn_reg, model: nn.Module,
                 w_b=1, w_m=1, w_g=1, w_mass=1, w_reg=1, w_limit=1, max_allowed=4,
                 tm_idxs = None, tkm_idxs = None, dim = 2, use_jacobi_distance = True,
                 normalization_mode = "time"):
        """Store loss functions, loss weights, and intermediate-marginal indices.

        Args:
            dataset: Dataset providing per-timestep particle batches.
            loss_fn: Sinkhorn loss (p=2) used for marginal/energy position terms.
            loss_fn_reg: Sinkhorn loss (p=1) used as a regularizer on marginals.
            model: The NeuralMap being trained.
            w_b, w_m, w_g, w_mass, w_reg, w_limit: Loss term weights.
            max_allowed: Unused mass-clamping bound, kept for config compatibility.
            tm_idxs: Reference-marginal indices to supervise.
            tkm_idxs: Integration-node indices matching tm_idxs.
            dim: Spatial dimension used when slicing reference marginals.
            use_jacobi_distance: Scale transport energy by squared mass drift (contact
                distance) if True, else use plain Wasserstein energy.
            normalization_mode: sum_normalization mode applied to the distance term.
        """
        super().__init__(model, dataset)
        self.T_theta = model
        self.loss_fn = loss_fn
        self.loss_fn_reg = loss_fn_reg
        self.w_b = w_b
        self.w_m = w_m
        self.w_g = w_g
        self.w_mass = w_mass
        self.w_reg = w_reg
        self.w_limit = w_limit
        self.max_allowed = max_allowed
        self.tm_idxs = list(tm_idxs) if tm_idxs is not None else []
        self.tkm_idxs = list(tkm_idxs) if tkm_idxs is not None else []
        self.dim = dim
        self.use_jacobi_distance = use_jacobi_distance
        self.normalization_mode = normalization_mode

    def get_loss(self, xm_refs, plot=False, horizon=0, alpha = torch.tensor([0.5])) -> torch.Tensor:
        """Compute loss between T_theta(x_in) and the stacked reference marginals.

        Args:
            xm_refs: Batch from sample_batch(), (data_list, mass_list).
            plot: If True, print per-marginal loss diagnostics.
            horizon: Current training horizon, used to weight intermediate marginals.
            alpha: Sigmoid sharpness for the (currently unused) horizon weighting.

        Returns:
            Tuple (loss, weighted_energy, weighted_marginals, zeros) of scalar tensors.
        """
        x_refs = xm_refs[0]
        m_refs = xm_refs[-1]
        xm_pred, drift, z_opt, growth = self.T_theta(x_refs[0], return_details=True)
        x_pred = xm_pred[..., :-1]
        m_pred = xm_pred[..., -1]
        n_nodes, n_samples, _ = x_pred.shape
        device = x_pred.device
        inter_marginals = torch.zeros(1, device=device)

        x_dist = x_pred.clone()
        x_dist[-1, :] = x_dist[-1, :].detach()

        for j in range(n_nodes):
            # if j is in the set of t_{km} associated to an intermediate marginal...
            if j in self.tkm_idxs:
                # then check which marginal it is
                m_idx = self.tkm_idxs.index(j)
                ref_idx = self.tm_idxs[m_idx]
                # compute the loss
                weights_pred = m_pred[j] / torch.sum(m_pred[j])
                weights_ref = torch.ones(x_refs[ref_idx].shape[0]).to(x_pred.device) / x_refs[ref_idx].shape[0]
                inter_loss_w2 = self.loss_fn(weights_pred, x_pred[j], weights_ref, x_refs[ref_idx][:, :self.dim])
                inter_loss_w1 = self.loss_fn_reg(weights_pred, x_pred[j], weights_ref, x_refs[ref_idx][:, :self.dim])
                inter_loss_mass = (torch.sum(m_pred[j]) / n_samples -
                                   (m_refs[ref_idx] / m_refs[0])).pow(2)
                individual_marginal_loss = inter_loss_w2 + self.w_reg * inter_loss_w1 + self.w_mass * inter_loss_mass

                weight = torch.sigmoid(alpha * (self.tkm_idxs[0] + horizon - j)).to(device)
                if plot:
                    print(f"individual_marginal_loss {j}: {individual_marginal_loss} (weight {weight})")
                if j == n_nodes - 1:
                    w_j = self.w_b
                else:
                    w_j = 1.
                inter_marginals += individual_marginal_loss * w_j
                # make the intermediate marginal prediction a fixed point of the geodesic
                x_dist[j, :] = x_dist[j, :].detach()

        distance = ((x_dist[1:, :] - x_dist[:-1, :])**2).sum(dim=-1)
        loss_drift = (self.w_limit / self.w_g) * m_pred.var(dim=-1).mean()
        drift = sum_normalization(drift, "time")
        distance = sum_normalization(distance, self.normalization_mode)
        drift_pow = drift.pow(2).squeeze()
        if self.use_jacobi_distance:
            jacobi_distance = drift_pow * distance
            z_loss = F.mse_loss(z_opt[1:], growth)  # scale tuned for the ablation study
            # z_loss = 0.
        else:
            # Ablation: plain Wasserstein energy, without scaling by the squared mass drift.
            jacobi_distance = distance
            z_loss = 0.
        energy = (m_pred[1:, :] * jacobi_distance).mean() + loss_drift + z_loss
        loss = self.w_g * energy + self.w_m * inter_marginals
        return loss, self.w_g * energy, self.w_m * inter_marginals, torch.zeros_like(inter_marginals)


def sum_normalization(x, mode="sample", eps=1e-3):
    """Rescale drift/distance (both shape (n_steps, n_samples)) before they
    enter the energy term.

    Args:
        x: Tensor to normalize, shape (n_steps, n_samples).
        mode: Normalization axis, see below.
        eps: Minimum row/column sum, to avoid division by zero.

    Returns:
        Tensor of the same shape as x.

    mode="sample" (default): normalize each timestep's row across the batch
        -- stabilizes scale over training while preserving each particle's
        total cost across time.
    mode="time": normalize each particle's column across time (legacy) --
        forces every particle's total cost to the same constant, erasing
        the per-particle/per-cluster cost differences (e.g. a far target
        needing more transport than a near one) that the mass-drift
        coupling is supposed to act on.
    mode="none": no normalization.
    """
    if mode == "none":
        return x
    if mode == "sample":
        dim = 1
    elif mode == "time":
        dim = 0
    else:
        raise ValueError(f"Unknown normalization mode: {mode!r}")
    n = x.shape[dim]
    total = x.sum(dim=dim, keepdim=True).clamp_min(eps)
    x = x * (n / total)
    return x
