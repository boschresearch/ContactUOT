"""
Neural components for parameterizing transport maps (geodesics) in Wasserstein dynamical systems.
"""

from typing import Type
import torch
from torch import nn


def build_mlp(dims: list[int], activation: Type[nn.Module] = nn.SiLU) -> nn.Sequential:
    """Construct a standard multilayer perceptron.

    Args:
        dims (list[int]):
            Sequence of layer sizes (including input and output dimensions).
        activation (Type[nn.Module]):
            Activation module applied between hidden layers.

    Returns:
        nn.Sequential: The assembled feedforward network."""
    mlp = []
    for idx in range(len(dims) - 1):
        mlp.append(torch.nn.Linear(dims[idx], dims[idx + 1]))
        if idx < len(dims) - 2:
            # Adding LayerNorm helps prevent mode collapse in vector fields
            mlp.append(nn.LayerNorm(dims[idx + 1]))
            mlp.append(activation())
    return nn.Sequential(*mlp)


class TrunkWithTimeFiLM(nn.Module):
    """
    Trunk where time FiLM is applied after every hidden layer,
    giving time full influence over the representation at each depth.
    """
    def __init__(self, input_dim, hidden, layers, time_dim):
        """Build one Linear+LayerNorm block and one time-FiLM MLP per hidden layer.

        Args:
            input_dim: Input feature dimension.
            hidden: Hidden width of every layer.
            layers: Number of hidden layers.
            time_dim: Dimension of the (already Fourier-encoded) time input.
        """
        super().__init__()

        # One linear + LayerNorm per hidden layer (no activation — FiLM replaces it)
        self.linears = nn.ModuleList()
        self.norms   = nn.ModuleList()
        self.linears.append(nn.Linear(input_dim, hidden))
        self.norms.append(nn.LayerNorm(hidden))
        for _ in range(layers - 1):
            self.linears.append(nn.Linear(hidden, hidden))
            self.norms.append(nn.LayerNorm(hidden))

        # One FiLM MLP per layer — each learns its own time modulation
        self.time_films = nn.ModuleList([
            build_mlp([time_dim, hidden, 2 * hidden], nn.SiLU)
            for _ in range(layers)
        ])

        self.activation = nn.SiLU()

    def forward(self, x_encoded, t_encoded):
        """Run x_encoded (B, input_dim) through the trunk, FiLM-modulated by t_encoded (B,).

        Args:
            x_encoded: Input features, shape (B, input_dim).
            t_encoded: Fourier-encoded time, shape (B, time_dim).

        Returns:
            torch.Tensor: Trunk output, shape (B, hidden).
        """
        h = x_encoded
        for i, (linear, norm, time_film) in enumerate(zip(self.linears, self.norms, self.time_films)):
            h = linear(h)
            h = norm(h)
            gamma, beta = time_film(t_encoded).chunk(2, dim=-1)
            h = (1 + gamma) * h + beta
            if i < len(self.linears) - 1:      # no activation on the last layer
                h = self.activation(h)

        return h

class VectorFieldWithTime(nn.Module):
    """Time-conditioned vector field predicting both drift (dz) and log-growth-rate."""
    def __init__(self, dim, hidden=128, layers=4, num_frequencies=3, interFiLM=False,
                 logm_time_only=False):
        """Build the shared trunk and the drift/growth heads.

        Args:
            dim: Spatial dimension of the state.
            hidden: Hidden width of the trunk and heads.
            layers: Number of hidden layers in the trunk.
            num_frequencies: Number of Fourier frequency bands for positional encoding.
            interFiLM: If True, use TrunkWithTimeFiLM (time FiLM at every layer)
                instead of concatenating Fourier-encoded time to the input.
            logm_time_only: If True, the growth-rate head only sees encoded time, not space.
        """
        super().__init__()
        self.num_frequencies = num_frequencies
        self.interFiLM = interFiLM
        self.logm_time_only = logm_time_only

        if not interFiLM:
            input_dim = dim + 1
            encoded_dim = input_dim + 2 * input_dim * self.num_frequencies
            self.shared_trunk = build_mlp([encoded_dim] + [hidden]*layers, nn.SiLU)
        else:
            time_dim = 1 + 2 * self.num_frequencies
            self.shared_trunk = TrunkWithTimeFiLM(
                input_dim=dim,
                hidden=hidden,
                layers=layers,
                time_dim=time_dim
            )

        # separate heads
        self.vel_head = nn.Linear(hidden, dim)
        if self.logm_time_only:
            time_encoded_dim = 1 + 2 * self.num_frequencies
            self.logm_head_time = build_mlp([time_encoded_dim, hidden, 1], nn.SiLU)
        else:
            self.logm_head = nn.Linear(hidden, 1)

    def fourier_encode(self, x):
        """Concatenate x with its sin/cos Fourier features at num_frequencies octaves.

        Args:
            x: Input tensor, shape (..., d).

        Returns:
            torch.Tensor: shape (..., d * (1 + 2 * num_frequencies)).
        """
        out = [x]
        for i in range(self.num_frequencies):
            freq = 2.0 ** i
            out.append(torch.sin(x * freq))
            out.append(torch.cos(x * freq))
        return torch.cat(out, dim=-1)

    def forward(self, x, t):
        """Predict drift and log-growth-rate at state x and time t.

        Args:
            x: (B, dim) or (B, T, dim)
            t: scalar, (B,), or (B, T)

        Returns:
            Tuple (dz, logm_dot) of drift and log-growth-rate, both flattened over
            any leading (B, T) dims.
        """
        if x.dim() == 3:
            B, T, dim = x.shape
            if t.dim() == 1:
                t = t.unsqueeze(0).expand(B, T)
            elif t.dim() == 2:
                pass
            else:
                raise ValueError(f"Invalid t shape: {t.shape}")
            x = x.reshape(B * T, dim)
            t = t.reshape(B * T)
        else:
            B = x.shape[0]
            t = torch.full((B,), t, device=x.device, dtype=x.dtype)

        t_encoded = self.fourier_encode(t.unsqueeze(-1))
        if not self.interFiLM:
            x_t = torch.cat([x, t.unsqueeze(-1)], dim=1)
            encoded = self.fourier_encode(x_t)
            h = self.shared_trunk(encoded)
        else:
            h = self.shared_trunk(x, t_encoded)

        dz = self.vel_head(h)
        if self.logm_time_only:
            logm_dot = self.logm_head_time(t_encoded)
        else:
            logm_dot = self.logm_head(h)
        return dz, logm_dot


class NeuralMap(nn.Module):
    """
    Explicit Euler integration of a learned vector field.

    z_{k+1} = z_k + tau * f(z_k, t_k)
    m_{k+1} = m_k + tau * g(z_k, t_k)
    """

    def __init__(
        self,
        dim: int,
        n_steps: int = 10,
        n_features: int = 128,
        n_layers: int = 4,
        n_frequencies: int = 4,
        t0: float = 0.0,
        t1: float = 1.0,
        experiment_name: str = "default",
        logm_time_only: bool = False
    ):
        """Build the vector field, selecting architecture/conditioning by experiment_name.

        Args:
            dim: Spatial dimension of the state (excluding mass).
            n_steps: Number of Euler integration steps.
            n_features: Hidden width of the vector field.
            n_layers: Number of hidden layers in the vector field's trunk.
            n_frequencies: Number of Fourier frequency bands for positional encoding.
            t0, t1: Integration start/end times.
            experiment_name: Selects architecture variant ("ucf101_dc" uses the
                interleaved-FiLM trunk).
            logm_time_only: If True, the growth-rate head only sees encoded time.
        """
        super().__init__()

        self.dim = dim
        self.n_steps = n_steps
        self.t0 = t0
        self.t1 = t1
        self.tau = torch.tensor((self.t1 - self.t0) / self.n_steps)
        self.experiment_name = experiment_name

        if experiment_name == "ucf101_dc":
            print("Architecture based on the Interleaved FiLM")
            interFiLM = True
        else:
            print("Architecture based on MLP")
            interFiLM = False

        self.vector_field = VectorFieldWithTime(
            dim=dim,
            hidden=n_features,
            layers=n_layers,
            num_frequencies=n_frequencies,
            interFiLM=interFiLM,
            logm_time_only=logm_time_only
        )

    def forward(self, z_m0: torch.Tensor, return_details=False, return_growth=False):
        """Integrate the vector field forward from z_m0 for n_steps Euler steps.

        Args:
            z_m0: Initial (position, mass) state, shape (batch, dim+1).
            return_details: If True, also return (v_opt, S_opt, growth) from compute_v_S.
            return_growth: If True, also return the per-step log-growth rate.

        Returns:
            If return_details: (xm_traj, v_opt, S_opt, growth).
            Elif return_growth: (xm_traj, log_g), log_g shape (n_steps, batch).
            Else: xm_traj alone.
            xm_traj is the concatenated (position, mass) trajectory, shape
            (n_steps+1, batch, dim+1).
        """
        z = z_m0[:, :-1]
        m = z_m0[:, -1:]

        traj = [z]
        mass = [m]
        log_g = []

        times = torch.linspace(
            self.t0, self.t1, self.n_steps + 1, device=z.device
        )
        tau = self.tau.to(z_m0.device)

        for k in range(self.n_steps):
            t = times[k]

            dz, logm_dot = self.vector_field(z, t)
            z = z + tau * dz
            logm_dot = 10 * torch.tanh(logm_dot / 10)
            m = m * torch.exp(tau * logm_dot)

            traj.append(z)
            mass.append(m)
            log_g.append(logm_dot)

        traj = torch.stack(traj, dim=0)
        mass = torch.stack(mass, dim=0)
        if return_details:
            v_opt, z_opt, growth = self.compute_v_S_z(mass)
            return torch.cat([traj, mass], dim=-1), v_opt, z_opt, growth
        if return_growth:
            return torch.cat([traj, mass], dim=-1), torch.stack(log_g, dim=0).squeeze()

        return torch.cat([traj, mass], dim=-1)

    def compute_v_S(self, mass, log_g):
        """Recurrence-based (WFR) computation of the optimal velocity/action pair from mass and log_g.

        Args:
            mass: Predicted mass trajectory, shape (n_steps+1, batch, 1).
            log_g: Per-step log-growth rate, a list of n_steps tensors shape (batch, 1).

        Returns:
            Tuple (v_opt, S_opt, growth), each shape (n_steps, batch, 1).
        """
        tau = self.tau.to(mass.device)
        growth = torch.diff(mass, dim=0) / mass[:-1, :]
        delta_growth = torch.diff(growth, dim=0)

        v_list = [torch.ones_like(growth[0])]
        S_list = [torch.zeros_like(growth[0])]

        for j in range(growth.shape[0] - 1):
            v_prev = v_list[-1]
            S_prev = S_list[-1]

            v_next = v_prev * torch.exp(tau * log_g[j])
            v_next = v_next + tau * S_prev * delta_growth[j]
            S_next = S_prev * torch.exp(tau * log_g[j])
            S_next = S_next + tau * (0.5 * v_prev**2)

            v_list.append(v_next)
            S_list.append(S_next)

        return torch.stack(v_list, dim=0), torch.stack(S_list, dim=0), growth

    def compute_v_S_z(self, mass):
        """Jacobi/contact-distance recurrence for the optimal velocity, action z, and growth.

        Args:
            mass: Predicted mass trajectory, shape (n_steps+1, batch, 1).

        Returns:
            Tuple (v_opt, z_opt, growth): v_opt shape (n_steps, batch)
            (edge-indexed), z_opt shape (n_steps+1, batch) (node-indexed),
            growth shape (n_steps, batch).
        """
        wz = -1
        tau = self.tau.to(mass.device)
        growth = torch.diff(mass, dim=0) / (mass[:-1, :] * tau * wz)
        growth = growth.squeeze(-1)
        v = torch.ones_like(growth[0])
        S = torch.zeros_like(growth[0])
        z = growth[0].clone()
        v_list = [v]
        S_list = [S]
        z_list = [z]
        ref = 0.5 * v.pow(2) * mass[0].squeeze(-1)

        for k in range(growth.shape[0]):
            g = growth[k]
            if k < growth.shape[0]-1:
                dg = growth[k+1] - growth[k]
            else:
                dg = torch.zeros_like(g)

            v_prev, S_prev = v, S

            v = v_prev + tau * (1/(1+v_prev)) * (0.5*wz*g*v_prev + wz*dg*S_prev) # Eq. (8a)
            S = S_prev - 0.5*tau*v_prev.pow(2) + 0.5*wz*g*S_prev # Eq. (8b)
            z = z + 0.5*tau*v_prev.pow(2)*mass[k].squeeze(-1) + 0.5*tau*wz*g.pow(2) - tau*ref  # Eq. (8c)

            v_list.append(v)
            S_list.append(S)
            z_list.append(z)

        # v is edge-indexed (paired with the n_steps transitions, like `distance`
        # in training.py) so drop the trailing update that has no following edge;
        # z stays node-indexed (n_steps+1, like mass/traj) for the z_opt[1:] vs
        # growth comparison in training.py.
        return torch.stack(v_list[:-1], dim=0), torch.stack(z_list, dim=0), growth


    def compute_v_S_z_FR(self, mass):
        """Fisher-Rao-metric variant of compute_v_S_z (wz=+1, no S term).

        Args:
            mass: Predicted mass trajectory, shape (n_steps+1, batch, 1).

        Returns:
            Tuple (v_opt, z_opt, growth): v_opt shape (n_steps, batch)
            (edge-indexed), z_opt shape (n_steps+1, batch) (node-indexed),
            growth shape (n_steps, batch).
        """
        wz = 1
        tau = self.tau.to(mass.device)
        growth = torch.diff(mass, dim=0) / (mass[:-1, :] * tau * wz)
        growth = growth.squeeze(-1)
        v = torch.ones_like(growth[0])
        z = growth[0].clone()
        v_list = [v]
        z_list = [z]

        for k in range(growth.shape[0]):
            g = growth[k]
            if k < growth.shape[0]-1:
                dg = growth[k+1] - growth[k]
            else:
                dg = torch.zeros_like(g)

            v_prev, z_prev = v, z

            v = v_prev + tau * (1/(1+v_prev)) * (-wz*g*dg) # Eq. (8a)
            z = z_prev - 0.5*tau*dg.pow(2) - 0.5*wz*g.pow(2) # Eq. (45b)

            v_list.append(v)
            z_list.append(z)

        return torch.stack(v_list[:-1], dim=0), torch.stack(z_list, dim=0), growth
