"""Dataset classes exposing per-timestep particle batches for every experiment.

Each dataset implements a common interface: get_certain_time_data(t),
sample_batch(x, batch_size), sample_particles_batch(batch_size), and
get_all_particles_batch(), all keyed by the dataset's self.time_steps.
"""

import os
import numpy as np
import torch
import pandas as pd


class Gaussian2dDataset:
    """Toy 2D dataset: a small Gaussian at t=0 translated to (1, 1) at t=1."""

    def __init__(self, device, start_n_samples=1000, end_n_samples=2000):
        """Sample the t=0 and t=1 Gaussian point clouds.

        Args:
            device: Device to place the loaded tensors on.
            start_n_samples: Number of points to sample at t=0.
            end_n_samples: Number of points to sample at t=1.
        """
        self.device = device
        self.time_steps = np.array([0.0, 1.0])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

        self.data_time0 = 0.05 * np.random.randn(start_n_samples, 2)
        self.data_time1 = 0.05 * np.random.randn(end_n_samples, 2) + np.array([1.0, 1.0])

    def get_certain_time_data(self, t):
        """Return the point cloud at time t (0.0 or 1.0).

        Args:
            t: 0.0 or 1.0.

        Returns:
            np.ndarray: Points at time t.

        Raises:
            ValueError: If t is neither 0.0 nor 1.0.
        """
        if t == 0.0:
            return self.data_time0
        elif t == 1.0:
            return self.data_time1
        else:
            raise ValueError(f"Time t={t} not supported in the dataset.")

    def sample_batch(self, x, batch_size):
        """Randomly sample batch_size rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample.

        Returns:
            torch.Tensor: Sampled rows, shape (batch_size, dim).
        """
        indices = torch.randint(0, len(x), (batch_size,))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}
        batch0 = self.sample_batch(self.get_certain_time_data(0.0), batch_size)
        batch1 = self.sample_batch(self.get_certain_time_data(1.0), batch_size)
        particles_dict[0.0] = batch0.to(self.device)
        particles_dict[1.0] = batch1.to(self.device)
        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}
        batch0 = torch.tensor(self.get_certain_time_data(0.0), dtype=torch.float32)
        batch1 = torch.tensor(self.get_certain_time_data(1.0), dtype=torch.float32)
        particles_dict[0.0] = batch0.to(self.device)
        particles_dict[1.0] = batch1.to(self.device)
        return particles_dict


class MouseDataset():
    """Mouse hematopoiesis dataset (2D), loaded from datasets/mouse_hematopoiesis.csv."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "./datasets/mouse_hematopoiesis.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0.00, 1.00, 2.00])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t][["x1", "x2"]].values

    def sample_batch(self, x, batch_size):
        """Randomly sample batch_size rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample.

        Returns:
            torch.Tensor: Sampled rows, shape (batch_size, dim).
        """
        indices = torch.randint(0, len(x), (batch_size,))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def sample_validation_particles_batch(self, batch_size):
        """Sample a held-out batch at t=2.00 for validation. (Currently unused.)

        Args:
            batch_size: Unused; kept for interface compatibility.

        Returns:
            Tuple (validation_time, validation_batch): 2.00 and all particles at that time.
        """
        validation_batch = torch.tensor(self.get_certain_time_data(2.00), dtype=torch.float).to(self.device)
        validation_time = 2.00
        return (validation_time, validation_batch)


class SimulationDataset():
    """Synthetic gene-expression simulation dataset, loaded from datasets/simulation_gene_data.csv."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "datasets/simulation_gene_data.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0.00, 1.00, 2.00, 3.00, 4.00])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t][["x1", "x2"]].values

    def sample_batch(self, x, batch_size):
        """Randomly sample batch_size rows from x, without replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample.

        Returns:
            torch.Tensor: Sampled rows, shape (batch_size, dim).
        """
        indices = torch.randperm(len(x))[:batch_size]
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def remove_points_near_origin(self, radius=0.5):
        """Drop rows within radius of the origin from the underlying dataframe.

        Args:
            radius: Exclusion radius around the origin.

        Returns:
            None. Mutates self.df in place.
        """
        dist2 = self.df["x1"]**2 + self.df["x2"]**2
        self.df = self.df[dist2 > radius**2]


class AblationDataset():
    """Synthetic 2D ablation dataset, loaded from datasets/ablation.csv (see datasets/generate_ablation.py)."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "datasets/ablation.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0.00, 1.00, 2.00, 3.00, 4.00])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t][["x1", "x2"]].values

    def sample_batch(self, x, batch_size):
        """Randomly sample batch_size rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample.

        Returns:
            torch.Tensor: Sampled rows, shape (batch_size, dim).
        """
        indices = torch.randint(0, len(x), (batch_size,))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict


class EmtDataset():
    """Epithelial-mesenchymal transition dataset, loaded from datasets/emt.csv."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "datasets/emt.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0.00, 1.00, 2.00, 3.00])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t].iloc[:, 1:].values

    def sample_batch(self, x, batch_size):
        """Randomly sample batch_size rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample.

        Returns:
            torch.Tensor: Sampled rows, shape (batch_size, dim).
        """
        indices = torch.randint(0, len(x), (batch_size,))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict


class VeresDataset():
    """Veres pancreatic beta-cell differentiation dataset, loaded from datasets/Veres_alltime.csv."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "datasets/Veres_alltime.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0.00, 1.00, 2.00, 3.00, 4.00, 5.00, 6.00, 7.00])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t].iloc[:, 1:].values

    def sample_batch(self, x, batch_size):
        """Randomly sample batch_size rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample.

        Returns:
            torch.Tensor: Sampled rows, shape (batch_size, dim).
        """
        indices = torch.randint(0, len(x), (batch_size,))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict


class MouseHardDataset():
    """Weinreb mouse hematopoiesis dataset (UMAP-reduced), loaded from datasets/Weinreb_alltime_umap.csv."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "datasets/Weinreb_alltime_umap.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0.00, 1.00, 2.00])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t].iloc[:, 1:].values

    def sample_batch(self, x, batch_size):
        """Randomly sample min(batch_size, len(x)) rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample (clamped to len(x)).

        Returns:
            torch.Tensor: Sampled rows, shape (min(batch_size, len(x)), dim).
        """
        indices = torch.randint(0, len(x), (min(batch_size, len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict


class Dim50Dataset():
    """Weinreb mouse hematopoiesis dataset (50D), loaded from datasets/Weinreb_alltime.csv."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "datasets/Weinreb_alltime.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0.00, 1.00, 2.00])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t].iloc[:, 1:].values

    def sample_batch(self, x, batch_size):
        """Randomly sample min(batch_size, len(x)) rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample (clamped to len(x)).

        Returns:
            torch.Tensor: Sampled rows, shape (min(batch_size, len(x)), dim).
        """
        indices = torch.randint(0, len(x), (min(batch_size, len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict


class Gaussian20Dataset():
    """Synthetic 20D Gaussian dataset, loaded from datasets/gaussian_20d.csv."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "datasets/gaussian_20d.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0.00, 1.00])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t].iloc[:, 1:].values

    def sample_batch(self, x, batch_size):
        """Randomly sample min(batch_size, len(x)) rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample (clamped to len(x)).

        Returns:
            torch.Tensor: Sampled rows, shape (min(batch_size, len(x)), dim).
        """
        indices = torch.randint(0, len(x), (min(batch_size, len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict


class Gaussian50Dataset():
    """Synthetic 50D Gaussian dataset, loaded from datasets/gaussian_50d.csv."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "datasets/gaussian_50d.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0.00, 1.00])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t].iloc[:, 1:].values

    def sample_batch(self, x, batch_size):
        """Randomly sample min(batch_size, len(x)) rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample (clamped to len(x)).

        Returns:
            torch.Tensor: Sampled rows, shape (min(batch_size, len(x)), dim).
        """
        indices = torch.randint(0, len(x), (min(batch_size, len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict


class Gaussian100Dataset():
    """Synthetic 100D Gaussian dataset, loaded from datasets/gaussian_100d.csv."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "datasets/gaussian_100d.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0.00, 1.00])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t].iloc[:, 1:].values

    def sample_batch(self, x, batch_size):
        """Randomly sample min(batch_size, len(x)) rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample (clamped to len(x)).

        Returns:
            torch.Tensor: Sampled rows, shape (min(batch_size, len(x)), dim).
        """
        indices = torch.randint(0, len(x), (min(batch_size, len(x)),))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict


class ZebrafishDataset():
    """Zebrafish embryogenesis dataset, loaded from datasets/Zebrafish.csv."""

    def __init__(self, device):
        """Load the CSV and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.csv_path = "./datasets/Zebrafish.csv"
        self.df = pd.read_csv(self.csv_path)
        self.device = device
        self.time_steps = np.array([0., 1., 2., 3., 4., 5., 6.])
        self.relative_mass = None
        self.max_time = np.max(self.time_steps)

    def get_certain_time_data(self, t):
        """Return all particles at time t as a NumPy array.

        Args:
            t: Timestep value in self.time_steps.

        Returns:
            np.ndarray: Particles at time t.
        """
        return self.df[self.df["samples"] == t].iloc[:, 1:].values

    def sample_batch(self, x, batch_size):
        """Randomly sample batch_size rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample.

        Returns:
            torch.Tensor: Sampled rows, shape (batch_size, dim).
        """
        indices = torch.randint(0, len(x), (batch_size,))
        return torch.tensor(x[indices.numpy()], dtype=torch.float32)

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = self.sample_batch(self.get_certain_time_data(time), batch_size)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            this_batch = torch.tensor(self.get_certain_time_data(time), dtype=torch.float32)
            particles_dict[time] = this_batch.to(self.device)

        return particles_dict


FFHQ_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "ffhq",
)


def _get_ffhq_mask(gender, age, category):
    """Build a boolean mask selecting FFHQ samples matching category (MAN/WOMAN/ADULT/CHILDREN).

    Args:
        gender: Array of "male"/"female" labels.
        age: Array of ages (-1 for unknown).
        category: One of "MAN", "WOMAN", "ADULT", "CHILDREN".

    Returns:
        np.ndarray: Boolean mask, shape (N,).

    Raises:
        ValueError: If category is not recognized.
    """
    if category == "MAN":
        return (gender == "male").reshape(-1)
    elif category == "WOMAN":
        return (gender == "female").reshape(-1)
    elif category == "ADULT":
        return ((age >= 18) & (age != -1)).reshape(-1)
    elif category == "CHILDREN":
        return ((age < 18) & (age != -1)).reshape(-1)
    else:
        raise ValueError(f"Unknown category {category}")


class FFHQDataset():
    """FFHQ children->adult experiment using ALAE latents (data/latents.npy, dim=512)."""

    def __init__(self, device, input_data="CHILDREN", target_data="ADULT"):
        """Load ALAE latents and split them into input_data/target_data subsets.

        Args:
            device: Device to place the loaded tensors on.
            input_data: Category for the t=0 subset (see _get_ffhq_mask).
            target_data: Category for the t=1 subset (see _get_ffhq_mask).
        """
        latents = np.load(os.path.join(FFHQ_DATA_DIR, "latents.npy"))
        gender = np.load(os.path.join(FFHQ_DATA_DIR, "gender.npy"))
        age = np.load(os.path.join(FFHQ_DATA_DIR, "age.npy"))

        x_mask = _get_ffhq_mask(gender, age, input_data)
        y_mask = _get_ffhq_mask(gender, age, target_data)

        self.x_data = torch.tensor(latents[x_mask], dtype=torch.float32)
        self.y_data = torch.tensor(latents[y_mask], dtype=torch.float32)
        self.device = device
        self.relative_mass = None
        self.time_steps = np.array([0.0, 1.0])
        self.max_time = np.max(self.time_steps)

    def sample_batch(self, x, batch_size):
        """Randomly sample batch_size rows from x, with replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample.

        Returns:
            torch.Tensor: Sampled rows, shape (batch_size, dim).
        """
        indices = torch.randint(0, x.shape[0], (batch_size,))
        return x[indices]

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        data_train = {
            0: self.sample_batch(self.x_data, batch_size).to(self.device),
            1: self.sample_batch(self.y_data, batch_size).to(self.device)
        }
        return data_train

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        data_train = {
            0: self.x_data.to(self.device),
            1: self.y_data.to(self.device)
        }
        return data_train

    def get_particle_counts(self):
        """Population sizes without moving the (multi-GB at dim=9216) full
        dataset to GPU -- see BaseTrainer.sample_batch, which only needs
        these counts (for the true children/adult mass ratio), not the data
        itself.

        Args:
            None.

        Returns:
            dict[int, int]: {0: input population size, 1: target population size}.
        """
        return {0: self.x_data.shape[0], 1: self.y_data.shape[0]}

    def get_certain_time_data(self, t):
        """Return the input or target latent tensor at time t (0.0 or 1.0).

        Args:
            t: 0.0 or 1.0.

        Returns:
            torch.Tensor: Latents at time t.

        Raises:
            ValueError: If t is neither 0.0 nor 1.0.
        """
        if t == 0.0:
            return self.x_data
        elif t == 1.0:
            return self.y_data
        else:
            raise ValueError(f"Time t={t} not supported in the FFHQ dataset.")


def compute_psp_latent_stats(latents_flat, eps=1e-3):
    """Per-dimension mean/std of the raw (N, 9216) pSp W+ latents.

    The raw pSp W+ space has ~16x larger per-sample norm than the ALAE
    latent space and wildly non-uniform per-dimension variance (per-layer
    std ranges ~0.9-10.5, vs ALAE's near-uniform ~0.89 across all 512
    dims). Since ffhq_psp.py intentionally reuses every ffhq.py
    hyperparameter (entropy/blur, lr, w_limit, max_allowed, ...) for a
    clean encoder-only ablation, feeding in the raw pSp scale makes the
    Sinkhorn energy/mass-drift terms wildly mis-calibrated: training
    doesn't NaN outright (grad clipping catches that) but the loss spikes
    by up to 8 orders of magnitude mid-training and never reconverges.
    Standardizing per-dimension (same eps convention as sum_normalization
    above) fixes the scale without touching any hyperparameter, preserving
    the ablation's intent. decode_latents.py must invert this same
    transform (recomputed identically from psp_latent.npy) before
    decoding x_pred/x0 back into images.

    Args:
        latents_flat: Raw latents, shape (N, 9216).
        eps: Minimum allowed std, to avoid division by zero downstream.

    Returns:
        Tuple (mean, std), each shape (9216,).
    """
    mean = latents_flat.mean(axis=0)
    std = np.maximum(latents_flat.std(axis=0), eps)
    return mean, std


class FFHQPSPDataset(FFHQDataset):
    """Same FFHQ children->adult experiment, but in the pixel2style2pixel W+
    latent space (datasets/ffhq/psp_latent.npy, shape (N, 18, 512)) instead of
    the ALAE latent space. Used to ablate the framework w.r.t. the encoder.

    The (18, 512) per-layer StyleGAN style codes are flattened to a single
    9216-dim vector so they fit the existing flat-vector NeuralMap/dataset
    interface; decode_latents.py reshapes them back to (18, 512)
    before calling the pSp decoder. The vectors are standardized
    per-dimension before use -- see compute_psp_latent_stats() above.
    """

    def __init__(self, device, input_data="CHILDREN", target_data="ADULT"):
        """Load pSp latents, standardize them, and split into input_data/target_data subsets.

        Args:
            device: Device to place the loaded tensors on.
            input_data: Category for the t=0 subset (see _get_ffhq_mask).
            target_data: Category for the t=1 subset (see _get_ffhq_mask).
        """
        latents = np.load(os.path.join(FFHQ_DATA_DIR, "psp_latent.npy"))
        latents = latents.reshape(latents.shape[0], -1)
        self.psp_mean, self.psp_std = compute_psp_latent_stats(latents)
        latents = (latents - self.psp_mean) / self.psp_std
        gender = np.load(os.path.join(FFHQ_DATA_DIR, "gender.npy"))
        age = np.load(os.path.join(FFHQ_DATA_DIR, "age.npy"))

        x_mask = _get_ffhq_mask(gender, age, input_data)
        y_mask = _get_ffhq_mask(gender, age, target_data)

        self.x_data = torch.tensor(latents[x_mask], dtype=torch.float32)
        self.y_data = torch.tensor(latents[y_mask], dtype=torch.float32)
        self.device = device
        self.relative_mass = None
        self.time_steps = np.array([0.0, 1.0])
        self.max_time = np.max(self.time_steps)


UCF101_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "datasets", "ucf101",
)


class UCF101Dataset:
    """UCF101 breaststroke video-latent dataset, loaded from a precomputed .pt tensor."""

    def __init__(self, device):
        """Load the [N, T, D] latent tensor and set up the dataset's time grid.

        Args:
            device: Device to place the loaded tensors on.
        """
        self.data_path = os.path.join(UCF101_DATA_DIR, "ucf101_bs_latents_dc.pt")
        self.data = torch.load(self.data_path)  # [N, T, D]
        self.device = device
        self.N, self.T, self.D = self.data.shape
        self.time_steps = np.arange(self.T, dtype=float)
        self.max_time = np.max(self.time_steps)
        self.relative_mass = None

    def get_certain_time_data(self, t):
        """Return all N latents at frame index t, shape [N, D].

        Args:
            t: Frame index (0 to T-1).

        Returns:
            torch.Tensor: Latents at frame t, shape (N, D).
        """
        t = int(t)
        return self.data[:, t, :]

    def sample_batch(self, x, batch_size):
        """Randomly sample batch_size rows from x, without replacement.

        Args:
            x: Source array/tensor to sample from.
            batch_size: Number of rows to sample.

        Returns:
            torch.Tensor: Sampled rows, shape (batch_size, dim).
        """
        indices = torch.randperm(x.shape[0])[:batch_size]
        return x[indices]

    def sample_particles_batch(self, batch_size):
        """Sample batch_size particles at each timestep, as {time: tensor}.

        Args:
            batch_size: Number of particles to sample per timestep.

        Returns:
            dict[float, torch.Tensor]: {time: sampled particles}.
        """
        particles_dict = {}

        for time in self.time_steps:
            x = self.get_certain_time_data(time)
            batch = self.sample_batch(x, batch_size)
            particles_dict[time] = batch.to(self.device)

        return particles_dict

    def get_all_particles_batch(self):
        """Return every particle at every timestep, as {time: tensor}.

        Args:
            None.

        Returns:
            dict[float, torch.Tensor]: {time: all particles at that time}.
        """
        particles_dict = {}

        for time in self.time_steps:
            x = self.get_certain_time_data(time)
            particles_dict[time] = x.to(self.device)

        return particles_dict

class UCF101TiTokDataset(UCF101Dataset):
    """Same BreastStroke experiment as UCF101Dataset, but with frames encoded
    by the TiTok-S-128 tokenizer (Yu et al., 2024, "An Image is Worth 32
    Tokens for Reconstruction and Generation", NeurIPS) instead of the
    Chen et al. (2025) spatial encoder. TiTok flattens each frame into
    128 latent tokens x 12 dims (1536-dim total) with no 2D grid structure,
    vs. the original 32x8x8 (2048-dim) spatial latent.
    """
    def __init__(self, device):
        self.data_path = os.path.join(UCF101_DATA_DIR, "ucf101_bs_latents_titok.pt")
        self.data = torch.load(self.data_path)  # [N, T, D]
        self.device = device
        self.N, self.T, self.D = self.data.shape
        self.time_steps = np.arange(self.T, dtype=float)
        self.max_time = np.max(self.time_steps)
        self.relative_mass = None