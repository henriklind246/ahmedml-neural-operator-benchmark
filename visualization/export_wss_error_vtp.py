"""Write WSS-magnitude truth, predictions, and absolute errors onto AhmedML surface meshes for ParaView.

Reads the per-cell fields that `evaluation/eval_ahmedml.py --save_predictions`
saved for each model (full-geometry predictions, matching the reported metrics)
and writes one VTP per run with these cell arrays:
- wss_magnitude_true
- <model>_wss_magnitude_pred
- <model>_wss_magnitude_abs_error   (| |tau_pred| - |tau_true| |)

Saved predictions are in stored part order: preprocessing split each mesh into
stride-sampled parts (part k = cells[k::num_parts]) and the evaluator
concatenated them by part index. This script undoes that striding, then checks
that the saved truth matches the mesh's own pMean and wallShearStressMean cell
by cell, so a wrong ordering or a mismatched mesh fails instead of plotting
scrambled fields.

Needs NumPy and PyVista (the preprocessing environment), no GPU or model code.
"""

import argparse
import json
import os
import re

import numpy as np


TRUTH_FIELDS = ("pMean", "wallShearStressMean")


def mesh_order(n_cells, num_parts):
    """Original mesh cell index of each row of a saved prediction file."""
    return np.concatenate([np.arange(k, n_cells, num_parts) for k in range(num_parts)])


def to_mesh_order(fields, num_parts):
    """Reorder rows saved in stored part order back to mesh cell order."""
    out = np.empty_like(fields)
    out[mesh_order(len(fields), num_parts)] = fields
    return out


def count_parts(data_dir, run):
    """Number of stored parts for a run, counted the same way as the evaluation data loader."""
    pattern = re.compile(rf"{re.escape(run.replace('run', 'boundary'))}_points_part\d+\.npy")
    return sum(1 for name in os.listdir(os.path.join(data_dir, run)) if pattern.fullmatch(name))


def check_truth(true, mesh_fields):
    """Raise unless saved truth [p, wss_x, wss_y, wss_z] matches the mesh's own fields."""
    expected = np.column_stack([np.asarray(mesh_fields[name], dtype=np.float64) for name in TRUTH_FIELDS])
    if expected.shape != true.shape:
        raise RuntimeError(f"Mesh fields have shape {expected.shape}, saved truth {true.shape}")
    # Normalizing and denormalizing in float32 changes values only at round-off level.
    scale = np.abs(expected).max(axis=0)
    if not np.allclose(true, expected, rtol=1e-4, atol=1e-4 * scale):
        raise RuntimeError("Saved truth does not match the mesh's pMean/wallShearStressMean after reordering; "
                           "check --raw_dir and --num_parts")


def wss_arrays(true, predictions):
    """Return {array name: per-cell values} for the true |WSS| and each model's prediction and error."""
    true_magnitude = np.linalg.norm(true[:, 1:4], axis=1)
    arrays = {"wss_magnitude_true": true_magnitude}
    for model, pred in predictions.items():
        magnitude = np.linalg.norm(pred[:, 1:4], axis=1)
        arrays[f"{model}_wss_magnitude_pred"] = magnitude
        arrays[f"{model}_wss_magnitude_abs_error"] = np.abs(magnitude - true_magnitude)
    return {name: values.astype(np.float32) for name, values in arrays.items()}


def load_evaluation(directory, runs):
    """Return (model name, summary) for one evaluation directory that saved predictions for every run."""
    with open(os.path.join(directory, "test_summary.json")) as f:
        summary = json.load(f)
    model = summary["model"]
    missing = [run for run in runs
               if not os.path.isfile(os.path.join(directory, "predictions", f"{run}.npz"))]
    if missing:
        raise SystemExit(f"{directory} has no saved predictions for {' '.join(missing)}. Re-evaluate those runs:\n"
                         f"  sbatch slurm/eval_full_geometry.slurm {model} --runs {' '.join(runs)} --save_predictions")
    return model, summary


def export_run(run, evaluations, raw_dir, out_dir, num_parts=None):
    """Write <out_dir>/<run>.vtp and return its arrays."""
    import pyvista as pv

    vtp_path = os.path.join(raw_dir, run, f"{run.replace('run', 'boundary')}.vtp")
    mesh = pv.read(vtp_path)
    if num_parts is None:
        num_parts = count_parts(next(iter(evaluations.values()))["data_dir"], run)
        if num_parts == 0:
            raise SystemExit(f"Found no stored parts for {run}; pass --num_parts (preprocessing default: 20)")

    true, predictions = None, {}
    for model, summary in evaluations.items():
        with np.load(os.path.join(summary["eval_dir"], "predictions", f"{run}.npz")) as saved:
            if len(saved["true"]) != mesh.n_cells:
                raise RuntimeError(f"{model} saved {len(saved['true'])} cells for {run}; {vtp_path} has {mesh.n_cells}")
            model_true = to_mesh_order(saved["true"].astype(np.float64), num_parts)
            predictions[model] = to_mesh_order(saved["pred"].astype(np.float64), num_parts)
        if true is None:
            true = model_true
            check_truth(true, mesh.cell_data)
        elif not np.allclose(model_true, true, rtol=1e-5, atol=1e-5 * np.abs(true).max(axis=0)):
            raise RuntimeError(f"{model} saved different truth for {run} than {next(iter(evaluations))}")

    arrays = wss_arrays(true, predictions)
    out = mesh.copy()
    out.clear_data()
    for name, values in arrays.items():
        out.cell_data[name] = values
    out.save(os.path.join(out_dir, f"{run}.vtp"))
    return arrays


def main():
    parser = argparse.ArgumentParser(description="Export WSS-magnitude errors on AhmedML meshes as VTP.")
    parser.add_argument("eval_dirs", nargs="+",
                        help="Evaluation directories with saved predictions, one per model")
    parser.add_argument("--runs", nargs="+", required=True, help="Runs to export, e.g. from select_cases.py")
    parser.add_argument("--raw_dir", required=True, help="Raw AhmedML data root holding run_N/boundary_N.vtp")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_parts", type=int,
                        help="Stored parts per run (default: counted in each evaluation's data_dir)")
    args = parser.parse_args()

    evaluations = {}
    for directory in args.eval_dirs:
        model, summary = load_evaluation(directory, args.runs)
        if model in evaluations:
            raise SystemExit(f"{model} appears twice")
        # Read predictions next to the summary so copied evaluation directories still work.
        summary["eval_dir"] = directory
        evaluations[model] = summary

    os.makedirs(args.out_dir, exist_ok=True)
    for run in args.runs:
        arrays = export_run(run, evaluations, args.raw_dir, args.out_dir, args.num_parts)
        truth = arrays["wss_magnitude_true"]
        errors = np.concatenate([v for name, v in arrays.items() if name.endswith("_abs_error")])
        print(f"{os.path.join(args.out_dir, run + '.vtp')}  ({len(truth):,} cells)")
        print(f"  wss_magnitude_true 1st-99th percentile: {np.percentile(truth, 1):.4g} - {np.percentile(truth, 99):.4g}")
        print(f"  shared error color range (0 - 99th percentile over all models): 0 - {np.percentile(errors, 99):.4g}")


if __name__ == "__main__":
    main()
