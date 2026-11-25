"""Lookup tables mapping an experiment name to its dataset class and highlight times."""

import torch
from dataset import (
    MouseDataset, SimulationDataset, EmtDataset, VeresDataset, ZebrafishDataset,
    FFHQDataset, FFHQPSPDataset, UCF101Dataset, UCF101TiTokDataset, AblationDataset
)

DATA_LOADERS = {
    "simulation": SimulationDataset,
    "mouse": MouseDataset,
    "emt": EmtDataset,
    "veres": VeresDataset,
    "zebrafish": ZebrafishDataset,
    "ffhq_alae": FFHQDataset,
    "ffhq_psp": FFHQPSPDataset,
    "ucf101_dc": UCF101Dataset,
    "ucf101_titok": UCF101TiTokDataset,
    "ablation": AblationDataset,
}


def load_dataset(experiment_name: str, device: torch.device):
    """Instantiate the dataset class registered for experiment_name.

    Args:
        experiment_name: Key into DATA_LOADERS (case-insensitive).
        device: Device the dataset should place its tensors on.

    Returns:
        A dataset instance (see dataset.py).

    Raises:
        ValueError: If experiment_name is not a registered experiment.
    """
    try:
        dataset_cls = DATA_LOADERS[experiment_name.lower()]
    except KeyError as exc:
        raise ValueError(
            f"Unknown experiment '{experiment_name}'. "
            f"Available experiments: {list(DATA_LOADERS.keys())}"
        ) from exc

    return dataset_cls(device)


def dataset_highlight_times(experiment_name: str):
    """Return the reference-marginal times to highlight when plotting experiment_name.

    Args:
        experiment_name: One of the experiment keys in DATA_LOADERS.

    Returns:
        list[float]: Normalized (0..1) times to highlight.

    Raises:
        ValueError: If experiment_name is not recognized.
    """
    if experiment_name == "simulation":
        return [0, 0.25, 0.5, 0.75, 1.0]
    elif experiment_name == "mouse":
        return [0, 0.5, 1.0]
    elif experiment_name == "emt":
        return [0, 0.33, 0.66, 1.0]
    elif experiment_name == "veres":
        return [0, 0.142, 0.285, 0.428, 0.571, 0.714, 0.857, 1.0]
    elif experiment_name == "zebrafish":
        return [0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0]
    elif experiment_name == "ffhq_alae":
        return [0, 1.0]
    elif experiment_name == "ffhq_psp":
        return [0, 1.0]
    elif experiment_name == "ucf101_dc":
        return [0, 0.25, 0.5, 0.75, 1.0]
    elif experiment_name == "ucf101_titok":
        return [0, 0.25, 0.5, 0.75, 1.0]
    elif experiment_name == "ablation":
        return [0, 0.25, 0.5, 0.75, 1.0]
    else:
        raise ValueError(f"Unknown experiment '{experiment_name}'.")
