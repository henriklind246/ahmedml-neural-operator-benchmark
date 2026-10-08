import argparse
import csv
import json
import os
import pickle
import sys
import time
from collections import defaultdict

# Resolve project modules whether this script runs from the repo tree
# (evaluation/) or from a flat working directory, as on MSI.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.append(_p)

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import SequentialSampler
try:
    from evaluation.inference import iter_predictions
except ModuleNotFoundError:
    from inference import iter_predictions

try:
    from data.loaders.dataset_drivaerml_surface_numpy_chunk import (
        DrivAerChunkDataset,
        DrivAerMLVTUChunkDataLoader,
    )
except ImportError:
    from dataset.dataset_drivaerml_surface_numpy_chunk import (
        DrivAerChunkDataset,
        DrivAerMLVTUChunkDataLoader,
    )


MODEL_CONFIGS = {
    "transolver3_full": dict(
        n_hidden=256,
        n_layers=16,
        space_dim=6,
        fun_dim=0,
        n_head=8,
        mlp_ratio=2,
        out_dim=4,
        slice_num=64,
        unified_pos=0,
    ),
    "transolver3_efficient": dict(
        n_hidden=256,
        n_layers=8,
        space_dim=6,
        fun_dim=0,
        n_head=8,
        mlp_ratio=2,
        out_dim=4,
        slice_num=64,
        unified_pos=0,
    ),
    "linearno_full": dict(
        n_hidden=256,
        n_layers=16,
        space_dim=6,
        fun_dim=0,
        n_head=8,
        key_ratio=1,
        mlp_ratio=2,
        out_dim=4,
        slice_num=64,
        unified_pos=0,
    ),
    "linearno_efficient": dict(
        n_hidden=256,
        n_layers=8,
        space_dim=6,
        fun_dim=0,
        n_head=8,
        key_ratio=1,
        mlp_ratio=2,
        out_dim=4,
        slice_num=64,
        unified_pos=0,
    ),
    "lrsa_full": dict(
        n_hidden=232,
        n_layers=8,
        space_dim=6,
        fun_dim=0,
        n_head=8,
        mlp_ratio=1,
        out_dim=4,
        slice_num=64,
        unified_pos=0,
    ),
    "lrsa_efficient": dict(
        n_hidden=168,
        n_layers=8,
        space_dim=6,
        fun_dim=0,
        n_head=8,
        mlp_ratio=1,
        out_dim=4,
        slice_num=64,
        unified_pos=0,
    ),
}


EXPECTED_PARAMS = {
    "transolver3_full": 7_600_772,
    "transolver3_efficient": 3_868_740,
    "linearno_full": 7_567_108,
    "linearno_efficient": 3_851_908,
    "lrsa_full": 7_583_580,
    "lrsa_efficient": 4_029_340,
}


def build_model(model_name):
    kwargs = MODEL_CONFIGS[model_name]

    if model_name in {"transolver3_full", "transolver3_efficient"}:
        from models import Transolver_chunk_opt_matrix_mul
        return Transolver_chunk_opt_matrix_mul.Model(**kwargs)

    if model_name in {"linearno_full", "linearno_efficient"}:
        from models import LinearNO_chunk_opt_matrix_mul
        return LinearNO_chunk_opt_matrix_mul.Model(**kwargs)

    if model_name in {"lrsa_full", "lrsa_efficient"}:
        from models import LRSA_chunk_opt_matrix_mul
        return LRSA_chunk_opt_matrix_mul.Model(**kwargs)

    raise ValueError(f"Unknown model: {model_name}")


def load_state_dict(path, device):
    ckpt = torch.load(path, map_location=device)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state = ckpt["state_dict"]
    else:
        state = ckpt

    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    return state


def save_run_predictions(directory, run_name, truth_chunks, prediction_chunks):
    """Write one geometry's physical [p, wss_x, wss_y, wss_z] fields in stored part order."""
    np.savez(
        os.path.join(directory, f"{run_name}.npz"),
        true=np.concatenate(truth_chunks),
        pred=np.concatenate(prediction_chunks),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=sorted(MODEL_CONFIGS.keys()),
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--json_file", required=True)
    parser.add_argument("--norm_stats_file", required=True)
    parser.add_argument("--model_ckpt", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--inference_mode", choices=["chunkwise", "full_geometry"],
                        default="chunkwise", help="full_geometry shares context across every cell in a run")
    parser.add_argument("--chunk_size", type=int, default=50000,
                        help="Max cells per compute chunk in full_geometry mode (does not bound LRSA memory)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device, or cpu for small smoke tests")
    parser.add_argument("--runs", nargs="+", help="Optional subset of run names from the test split")
    parser.add_argument("--save_predictions", action="store_true",
                        help="Also write per-geometry true/predicted fields to OUT_DIR/predictions (~35 MB per geometry)")
    args = parser.parse_args()
    if args.chunk_size <= 0 or args.num_workers < 0:
        parser.error("--chunk_size must be positive and --num_workers nonnegative")

    import os
    os.makedirs(args.out_dir, exist_ok=True)
    prediction_dir = os.path.join(args.out_dir, "predictions")
    if args.save_predictions:
        os.makedirs(prediction_dir, exist_ok=True)

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)

    with open(args.json_file) as f:
        split = json.load(f)
    if not split["test_list"] or len(set(split["test_list"])) != len(split["test_list"]):
        raise ValueError("test_list must contain unique run names and must not be empty")
    if args.runs:
        unknown = set(args.runs) - set(split["test_list"])
        if unknown:
            raise ValueError(f"Requested runs are not in the test split: {sorted(unknown)}")
        split["test_list"] = [run for run in split["test_list"] if run in args.runs]

    with open(args.norm_stats_file, "rb") as f:
        norm_stats = pickle.load(f)

    label_mean_np = np.asarray(norm_stats["label_mean"], dtype=np.float64)
    label_std_np = np.asarray(norm_stats["label_std"], dtype=np.float64)
    if (label_mean_np.shape != (4,) or label_std_np.shape != (4,)
            or not np.isfinite(label_mean_np).all() or not np.isfinite(label_std_np).all()
            or (label_std_np <= 0).any()):
        raise ValueError("Normalization statistics must contain four finite means and positive stds")

    print("AhmedML label mean:", label_mean_np)
    print("AhmedML label std :", label_std_np)
    print("Test geometries   :", len(split["test_list"]))

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
        num_workers=args.num_workers,
    )

    print("Evaluation chunks :", len(loader))

    model = build_model(args.model).to(device)
    model.load_state_dict(load_state_dict(args.model_ckpt, "cpu"), strict=True)
    model.eval()

    num_params = sum(p.numel() for p in model.parameters())
    expected_params = EXPECTED_PARAMS[args.model]

    if num_params != expected_params:
        raise RuntimeError(
            f"Parameter-count mismatch for {args.model}: "
            f"expected {expected_params:,}, got {num_params:,}"
        )

    print("Model             :", args.model)
    print("Parameters        :", num_params)
    print("Inference mode    :", args.inference_mode)
    print("Compute chunk size:", args.chunk_size)

    # Score on CPU so full-geometry labels and float64 metrics do not consume VRAM.
    mean = torch.tensor(label_mean_np, dtype=torch.float32)
    std = torch.tensor(label_std_np, dtype=torch.float32)

    # Metrics retain only small per-run accumulators after each geometry is scored.
    stats = defaultdict(lambda: {
        "count": 0,
        "norm_sse": np.zeros(4, dtype=np.float64),
        "norm_abs": np.zeros(4, dtype=np.float64),
        "pressure_sse": 0.0,
        "pressure_den": 0.0,
        "wss_sse": 0.0,
        "wss_den": 0.0,
        "wss_vector_sse": 0.0,
        "wss_vector_den": 0.0,
    })

    # Geometries arrive contiguously, so saved fields hold at most one run in memory.
    saved_run, saved_true, saved_pred = None, [], []

    started = time.perf_counter()
    with torch.inference_mode():
        predictions = iter_predictions(model, loader, device, args.inference_mode, args.chunk_size)
        for pred, y, run_name in tqdm(predictions, desc="AhmedML test (output chunks)"):
            if not torch.isfinite(pred).all() or not torch.isfinite(y).all():
                raise RuntimeError(f"Non-finite predictions or labels for {run_name}")

            diff_norm = pred - y

            s = stats[run_name]

            # Number of surface cells in this chunk.
            n = y.shape[0] * y.shape[1]
            s["count"] += n

            s["norm_sse"] += (
                (diff_norm.double() ** 2)
                .sum(dim=(0, 1))
                .cpu()
                .numpy()
            )

            s["norm_abs"] += (
                diff_norm.abs()
                .double()
                .sum(dim=(0, 1))
                .cpu()
                .numpy()
            )

            # Convert normalized network outputs back to AhmedML physical units.
            y_phys = y * std + mean
            pred_phys = pred * std + mean

            if args.save_predictions:
                if run_name != saved_run:
                    if saved_run is not None:
                        save_run_predictions(prediction_dir, saved_run, saved_true, saved_pred)
                    saved_run, saved_true, saved_pred = run_name, [], []
                saved_true.append(y_phys.reshape(-1, 4).numpy())
                saved_pred.append(pred_phys.reshape(-1, 4).numpy())

            # Pressure.
            p_true = y_phys[..., 0]
            p_pred = pred_phys[..., 0]

            s["pressure_sse"] += (
                ((p_pred - p_true).double() ** 2).sum().item()
            )
            s["pressure_den"] += (
                (p_true.double() ** 2).sum().item()
            )

            # Full wall-shear-stress vector.
            wss_true_vec = y_phys[..., 1:4]
            wss_pred_vec = pred_phys[..., 1:4]

            s["wss_vector_sse"] += (
                ((wss_pred_vec - wss_true_vec).double() ** 2).sum().item()
            )
            s["wss_vector_den"] += (
                (wss_true_vec.double() ** 2).sum().item()
            )

            # Wall-shear-stress magnitude.
            wss_true = torch.linalg.vector_norm(wss_true_vec, dim=-1)
            wss_pred = torch.linalg.vector_norm(wss_pred_vec, dim=-1)

            s["wss_sse"] += (
                ((wss_pred - wss_true).double() ** 2).sum().item()
            )
            s["wss_den"] += (
                (wss_true.double() ** 2).sum().item()
            )

    if saved_run is not None:
        save_run_predictions(prediction_dir, saved_run, saved_true, saved_pred)
    elapsed = time.perf_counter() - started
    if set(stats) != set(split["test_list"]):
        raise RuntimeError("Evaluation did not cover every requested geometry")
    rows = []

    total_count = 0
    total_norm_sse = np.zeros(4, dtype=np.float64)
    total_norm_abs = np.zeros(4, dtype=np.float64)

    total_pressure_sse = 0.0
    total_pressure_den = 0.0
    total_wss_sse = 0.0
    total_wss_den = 0.0
    total_wss_vector_sse = 0.0
    total_wss_vector_den = 0.0

    for run_name in sorted(
        stats.keys(),
        key=lambda s: int(s.replace("run_", "").replace("run", ""))
    ):
        s = stats[run_name]

        mse = s["norm_sse"] / s["count"]
        mae = s["norm_abs"] / s["count"]

        pressure_l2re = np.sqrt(
            s["pressure_sse"] / max(s["pressure_den"], 1e-30)
        )

        wss_l2re = np.sqrt(
            s["wss_sse"] / max(s["wss_den"], 1e-30)
        )

        wss_vector_l2re = np.sqrt(
            s["wss_vector_sse"] / max(s["wss_vector_den"], 1e-30)
        )

        rows.append({
            "run": run_name,
            "num_cells": s["count"],
            "pressure_l2re": pressure_l2re,
            "wss_magnitude_l2re": wss_l2re,
            "wss_vector_l2re": wss_vector_l2re,
            "mse_pressure_norm": mse[0],
            "mse_wss_x_norm": mse[1],
            "mse_wss_y_norm": mse[2],
            "mse_wss_z_norm": mse[3],
            "mae_pressure_norm": mae[0],
            "mae_wss_x_norm": mae[1],
            "mae_wss_y_norm": mae[2],
            "mae_wss_z_norm": mae[3],
        })

        total_count += s["count"]
        total_norm_sse += s["norm_sse"]
        total_norm_abs += s["norm_abs"]

        total_pressure_sse += s["pressure_sse"]
        total_pressure_den += s["pressure_den"]
        total_wss_sse += s["wss_sse"]
        total_wss_den += s["wss_den"]
        total_wss_vector_sse += s["wss_vector_sse"]
        total_wss_vector_den += s["wss_vector_den"]

    csv_path = os.path.join(args.out_dir, "test_per_case.csv")

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    pressure_case = np.array([r["pressure_l2re"] for r in rows])
    wss_case = np.array([r["wss_magnitude_l2re"] for r in rows])
    wss_vector_case = np.array([r["wss_vector_l2re"] for r in rows])

    summary = {
        "model": args.model,
        "inference_mode": args.inference_mode,
        "chunk_size": args.chunk_size,
        "model_ckpt": os.path.abspath(args.model_ckpt),
        "data_dir": os.path.abspath(args.data_dir),
        "json_file": os.path.abspath(args.json_file),
        "norm_stats_file": os.path.abspath(args.norm_stats_file),
        "runs": split["test_list"],
        "predictions_dir": os.path.abspath(prediction_dir) if args.save_predictions else None,
        "device": str(device),
        "evaluation_seconds": elapsed,
        "peak_gpu_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None,
        "num_test_geometries": len(rows),
        "total_test_cells": int(total_count),

        "global_normalized_mse":
            (total_norm_sse / total_count).tolist(),

        "global_normalized_mae":
            (total_norm_abs / total_count).tolist(),

        "global_pressure_l2re":
            float(np.sqrt(total_pressure_sse / max(total_pressure_den, 1e-30))),

        "global_wss_magnitude_l2re":
            float(np.sqrt(total_wss_sse / max(total_wss_den, 1e-30))),

        "global_wss_vector_l2re":
            float(np.sqrt(total_wss_vector_sse / max(total_wss_vector_den, 1e-30))),

        "mean_case_pressure_l2re":
            float(pressure_case.mean()),

        "median_case_pressure_l2re":
            float(np.median(pressure_case)),

        "mean_case_wss_magnitude_l2re":
            float(wss_case.mean()),

        "median_case_wss_magnitude_l2re":
            float(np.median(wss_case)),

        "mean_case_wss_vector_l2re":
            float(wss_vector_case.mean()),

        "median_case_wss_vector_l2re":
            float(np.median(wss_vector_case)),
    }

    json_path = os.path.join(args.out_dir, "test_summary.json")

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== AhmedML TEST RESULTS ==========")
    print(f"Test geometries: {len(rows)}")
    print(f"Test cells:      {total_count:,}")

    print(
        f"\nGlobal pressure L2RE: "
        f"{100 * summary['global_pressure_l2re']:.3f}%"
    )

    print(
        f"Global WSS-mag L2RE:  "
        f"{100 * summary['global_wss_magnitude_l2re']:.3f}%"
    )

    print(
        f"Global WSS-vector L2RE: "
        f"{100 * summary['global_wss_vector_l2re']:.3f}%"
    )

    print(
        f"\nMean case pressure L2RE: "
        f"{100 * summary['mean_case_pressure_l2re']:.3f}%"
    )

    print(
        f"Mean case WSS-mag L2RE:  "
        f"{100 * summary['mean_case_wss_magnitude_l2re']:.3f}%"
    )

    print(
        f"Mean case WSS-vector L2RE: "
        f"{100 * summary['mean_case_wss_vector_l2re']:.3f}%"
    )

    print("\nNormalized MSE:", summary["global_normalized_mse"])
    print("Normalized MAE:", summary["global_normalized_mae"])

    print("\nSaved:")
    print(csv_path)
    print(json_path)
    if args.save_predictions:
        print(prediction_dir)


if __name__ == "__main__":
    main()
