import argparse
import json
import pickle
import os

import numpy as np
import pyvista as pv
import torch
from torch.utils.data import SequentialSampler

from dataset.dataset_drivaerml_surface_numpy_chunk import (
    DrivAerChunkDataset,
    DrivAerMLVTUChunkDataLoader,
)
from models import Transolver_chunk_opt_matrix_mul


MODEL_KWARGS = dict(
    n_hidden=256,
    n_layers=16,
    space_dim=6,
    fun_dim=0,
    n_head=8,
    mlp_ratio=2,
    out_dim=4,
    slice_num=64,
    unified_pos=0,
)


def load_state(path, device):
    state = torch.load(path, map_location=device)

    if "model_state_dict" in state:
        state = state["model_state_dict"]
    elif "state_dict" in state:
        state = state["state_dict"]

    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}

    return state


parser = argparse.ArgumentParser()
parser.add_argument("--run", required=True)
parser.add_argument("--data_dir", required=True)
parser.add_argument("--raw_dir", required=True)
parser.add_argument("--norm_stats_file", required=True)
parser.add_argument("--model_ckpt", required=True)
parser.add_argument("--output", required=True)
args = parser.parse_args()

device = torch.device("cuda:0")

with open(args.norm_stats_file, "rb") as f:
    stats = pickle.load(f)

mean = torch.tensor(
    np.asarray(stats["label_mean"]),
    dtype=torch.float32,
    device=device,
)

std = torch.tensor(
    np.asarray(stats["label_std"]),
    dtype=torch.float32,
    device=device,
)

split = {
    "train_list": [],
    "test_list": [args.run],
}

dataset = DrivAerChunkDataset(
    root=args.data_dir,
    data_list=split,
    train=False,
    label_fields=["pMean", "wallShearStressMean"],
    norm_stats_file=args.norm_stats_file,
)

loader = DrivAerMLVTUChunkDataLoader(
    dataset,
    batch_size=1,
    sampler=SequentialSampler(dataset),
    num_workers=4,
)

model = Transolver_chunk_opt_matrix_mul.Model(**MODEL_KWARGS).to(device)
model.load_state_dict(load_state(args.model_ckpt, device))
model.eval()

run_number = args.run.replace("run_", "").replace("run", "")

raw_vtp = os.path.join(
    args.raw_dir,
    args.run,
    f"boundary_{run_number}.vtp"
)

print("Loading:", raw_vtp)
mesh = pv.read(raw_vtp)

n_cells = mesh.n_cells
num_parts = len(dataset.run_chunks[args.run])

print("Cells:", n_cells)
print("Parts:", num_parts)

truth = np.empty((n_cells, 4), dtype=np.float32)
prediction = np.empty((n_cells, 4), dtype=np.float32)

with torch.no_grad():

    for part_idx, (x, y, _, _) in enumerate(loader):

        x = x.to(device)
        y = y.to(device)

        pred = model([x])[0]

        y_phys = y * std + mean
        pred_phys = pred * std + mean

        y_np = y_phys.squeeze(0).cpu().numpy()
        pred_np = pred_phys.squeeze(0).cpu().numpy()

        indices = np.arange(part_idx, n_cells, num_parts)

        if len(indices) != len(y_np):
            raise RuntimeError(
                f"Chunk {part_idx}: expected {len(indices)} cells "
                f"but got {len(y_np)}"
            )

        truth[indices] = y_np
        prediction[indices] = pred_np

        print(
            f"part {part_idx:02d}: "
            f"{len(indices):,} cells"
        )


# Pressure
mesh.cell_data["pressure_true"] = truth[:, 0]
mesh.cell_data["pressure_pred"] = prediction[:, 0]
mesh.cell_data["pressure_abs_error"] = np.abs(
    prediction[:, 0] - truth[:, 0]
)

# Full wall shear vectors
wss_true_vec = truth[:, 1:4]
wss_pred_vec = prediction[:, 1:4]

mesh.cell_data["wss_true_vector"] = wss_true_vec
mesh.cell_data["wss_pred_vector"] = wss_pred_vec

# Magnitudes
wss_true = np.linalg.norm(wss_true_vec, axis=1)
wss_pred = np.linalg.norm(wss_pred_vec, axis=1)

mesh.cell_data["wss_magnitude_true"] = wss_true
mesh.cell_data["wss_magnitude_pred"] = wss_pred
mesh.cell_data["wss_magnitude_abs_error"] = np.abs(
    wss_pred - wss_true
)

os.makedirs(os.path.dirname(args.output), exist_ok=True)
mesh.save(args.output)

print("\nSaved:")
print(args.output)
