"""CLI: train (or load) a NeuralMap on the chosen experiment, then visualize
and quantitatively evaluate the recovered trajectory."""

import os
import sys
import importlib
from datetime import datetime
from simple_parsing import ArgumentParser
import torch
import numpy as np

# main.py lives in CuOT/, but networks/, params/, datasets/, external/ stayed
# at the repo root -- add the repo root to sys.path so those still resolve
# as top-level packages regardless of where this script is invoked from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.architectures import NeuralMap
from networks.training import FitGeodesicProbability
from geomloss import SamplesLoss
from visualization import (
    visualizeresult, visualize_per_timestep, visualize_ablation_mode_leakage,
    visualizedatagrowth, visualize_drift_and_growth, compute_ablation_target_labels,
)
from quantitative_metrics import quantitative_test
from experiment_selector import load_dataset, dataset_highlight_times
from utils import logging

parser = ArgumentParser()
parser.add_argument('--params', type=str, help='Name of the params module inside params/')
parser.add_argument('--model_name', type=str, default=None)
parser.add_argument('--seed', type=int, default=None)
parser.add_argument('--resume', type=str, default=None,
                     help='Path to a checkpoint (saved every checkpoint_every epochs) to resume training from')
args = parser.parse_args()

module = importlib.import_module(f'params.{args.params}')
params = getattr(module, 'Params')

if args.model_name is None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    train = True
    os.makedirs("models", exist_ok=True)
else:
    timestamp = args.model_name
    train = False

if args.seed is None:
    seed = params.seed
else:
    seed = args.seed

torch.manual_seed(seed=seed)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = load_dataset(params.experiment, device)
x_dict = dataset.get_all_particles_batch()
x0_wo_mass = list(x_dict.values())[0][:params.visualization_batch]
x0 = torch.cat([x0_wo_mass, torch.ones_like(x0_wo_mass[:, :1])], dim=1)
logm_time_only = getattr(params, 'logm_time_only', False)
use_jacobi_distance = getattr(params, 'use_jacobi_distance', True)
model = NeuralMap(dim=params.dim, n_steps=params.n_steps,
                  n_features=params.n_features, n_layers=params.n_layers,
                  n_frequencies=params.n_frequencies, t1=params.t1,
                  experiment_name=params.experiment,
                  logm_time_only=logm_time_only)
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Trainable parameters: {num_params}")

MODEL_FILENAME = f"models/{params.experiment}_date_{timestamp}.pth"
CHECKPOINT_FILENAME = f"models/{params.experiment}_date_{timestamp}_checkpoint.pth"
PLOT_PATH = f"results/figures_{params.experiment}_date_{timestamp}"
os.makedirs(PLOT_PATH, exist_ok=True)

if train:
    LOSS_PATH = f"{PLOT_PATH}/loss_alg"
    loss_fn = SamplesLoss("sinkhorn", p=2, blur=params.entropy, scaling=params.scaling)
    loss_fn_reg = SamplesLoss("sinkhorn", p=1, blur=params.entropy, scaling=params.scaling)

    normalization_mode = getattr(params, 'normalization_mode', 'time')
    trainer = FitGeodesicProbability(dataset=dataset,
                                     loss_fn=loss_fn,
                                     loss_fn_reg=loss_fn_reg,
                                     model=model,
                                     w_b=params.w_b, w_m=params.w_m,
                                     w_g=params.w_g, w_mass = params.w_mass,
                                     w_reg = params.w_reg,
                                     w_limit = params.w_limit,
                                     max_allowed = params.max_allowed,
                                     tm_idxs = params.tm_idxs,
                                     tkm_idxs = params.tkm_idxs,
                                     dim=params.dim,
                                     use_jacobi_distance=use_jacobi_distance,
                                     normalization_mode=normalization_mode)

    losses = trainer.train(num_epochs=params.n_iterations, num_batches=params.n_batches,
                           batch_size=params.batch_size, n_steps=params.n_steps,
                           device=device, lr=params.lr, save_path=MODEL_FILENAME,
                           mode=0, loss_plot_name=LOSS_PATH, amp_enabled = params.amp_enabled,
                           checkpoint_path=CHECKPOINT_FILENAME, checkpoint_every=100,
                           resume_from=args.resume)
else:
    state_dict = torch.load(MODEL_FILENAME)
    model.load_state_dict(state_dict)
    model.to(device)


highlight_times = dataset_highlight_times(params.experiment)
with torch.no_grad():
    results, log_growth = model(x0, return_growth=True)
    xm_details, v_opt, z_opt, growth = model(x0, return_details=True)
x_pred = results[..., :-1]
print("x_pred: ", x_pred.shape)
np.save(f"{PLOT_PATH}/x_pred.npy", x_pred.cpu().detach().numpy())
np.save(f"{PLOT_PATH}/x0.npy", x0.cpu().detach().numpy())
RESULT_TRAJ_PATH = f"{PLOT_PATH}/result_traj"
RESULT_TIMESTEPS_PATH = f"{PLOT_PATH}/result_timesteps"
RESULT_LEAKAGE_PATH = f"{PLOT_PATH}/result_mode_leakage.pdf"
RESULT_GROWTH_PATH = f"{PLOT_PATH}/result_growth"
RESULT_METRIC_PATH = f"{PLOT_PATH}/result_metric"
RESULT_DRIFT_GROWTH_PATH = f"{PLOT_PATH}/result_drift_growth"

visualizeresult(dataset, results, RESULT_TRAJ_PATH, params.dim_reduction, highlight_times,
                visualization_batch=params.visualization_batch)
visualize_per_timestep(dataset, results, RESULT_TIMESTEPS_PATH, params.dim_reduction, highlight_times,
                       visualization_batch=params.visualization_batch)
if params.experiment == "ablation":
    leakage_info = visualize_ablation_mode_leakage(results, RESULT_LEAKAGE_PATH, tau=params.t1 / params.n_steps)
    print("mode leakage:", leakage_info)
visualizedatagrowth(dataset, model, RESULT_GROWTH_PATH, params.dim_reduction, params.dim, params.n_steps, params.t1)
if params.experiment == "ablation":
    cluster_labels, cluster_names = compute_ablation_target_labels(
        xm_details[-1, :, :-1].detach().cpu().numpy()
    )
    visualize_drift_and_growth(xm_details, v_opt, z_opt, growth, RESULT_DRIFT_GROWTH_PATH,
                               cluster_labels=cluster_labels, cluster_names=cluster_names)
pos_w2, mass = quantitative_test(dataset, results, p=2)
pos_w1, _ = quantitative_test(dataset, results, p=1)
logging(pos_w2, pos_w1, mass, seed=seed, save_path=RESULT_METRIC_PATH)
