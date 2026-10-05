"""Density parity plots of predicted against true surface pressure and |WSS|.

Reads an evaluation directory written by
`evaluation/eval_ahmedml.py --save_predictions` and bins every held-out cell
into a 2D histogram, so tens of millions of cells render without overplotting.
It uses only the saved fields, so it works for any model variant and needs only
NumPy and Matplotlib (no GPU or model code).

Each panel shows cell density on a log color scale, the y = x line, and the
median and 5-95% range of predictions within each true-value column. Read them as:
- median bending below/above y = x at the ends: extremes are under/overpredicted;
- slope < 1: the model compresses the output range;
- band width changing along x: heteroscedastic error.
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, LogNorm


QUANTITIES = {
    "pressure": ("Surface pressure", "p"),
    "wss": ("Wall shear stress magnitude", r"|\tau_w|"),
}

# Quantile curves are drawn only for true-value columns with enough cells.
MIN_COLUMN_CELLS = 100

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
AXIS = "#c3c2b7"
ACCENT = "#eb6834"
# Single-hue sequential ramp; the light end stays visible for one-cell bins.
DENSITY = LinearSegmentedColormap.from_list(
    "density", ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"]
)


def load_quantities(path):
    """Return {quantity: (true, predicted)} for one saved geometry."""
    with np.load(path) as fields:
        true, pred = fields["true"], fields["pred"]
    return {
        "pressure": (true[:, 0], pred[:, 0]),
        "wss": (np.linalg.norm(true[:, 1:4], axis=1), np.linalg.norm(pred[:, 1:4], axis=1)),
    }


def accumulate(paths, bins):
    """Stream saved geometries into per-quantity 2D histograms and error sums.

    The first pass finds one square range covering every true and predicted
    value, so extreme cells are never clipped; the second pass bins all cells.
    counts[i, j] holds cells with true value in bin i and prediction in bin j.
    """
    lo = dict.fromkeys(QUANTITIES, np.inf)
    hi = dict.fromkeys(QUANTITIES, -np.inf)
    for path in paths:
        for name, (true, pred) in load_quantities(path).items():
            lo[name] = min(lo[name], float(true.min()), float(pred.min()))
            hi[name] = max(hi[name], float(true.max()), float(pred.max()))

    results = {}
    for name in QUANTITIES:
        if not hi[name] > lo[name]:
            raise ValueError(f"{name} has no spread to plot")
        results[name] = {
            "edges": np.linspace(lo[name], hi[name], bins + 1),
            "counts": np.zeros((bins, bins), dtype=np.int64),
            # n, sum(t), sum(p), sum(t^2), sum(t*p), sum((p - t)^2)
            "sums": np.zeros(6, dtype=np.float64),
        }

    for path in paths:
        for name, (true, pred) in load_quantities(path).items():
            r = results[name]
            r["counts"] += np.histogram2d(true, pred, bins=(r["edges"], r["edges"]))[0].astype(np.int64)
            t, p = true.astype(np.float64), pred.astype(np.float64)
            r["sums"] += [t.size, t.sum(), p.sum(), t @ t, t @ p, (p - t) @ (p - t)]

    for name, r in results.items():
        if r["counts"].sum() != r["sums"][0]:
            raise RuntimeError(f"{name} histogram lost cells")
    return results


def metrics(sums):
    """Global relative L2 error (as in eval_ahmedml.py), R^2, and OLS slope of prediction on truth."""
    n, st, sp, stt, stp, sse = sums
    sst = stt - st * st / n
    return {
        "cells": int(n),
        "rel_l2": float(np.sqrt(sse / stt)),
        "r2": float(1.0 - sse / sst),
        "slope": float((stp - st * sp / n) / sst),
    }


def conditional_quantiles(counts, edges, quantiles=(0.05, 0.5, 0.95)):
    """Quantiles of the prediction within each true-value column (NaN for sparse columns)."""
    out = np.full((len(quantiles), counts.shape[0]), np.nan)
    cdf = np.cumsum(counts, axis=1)
    for i in np.flatnonzero(cdf[:, -1] >= MIN_COLUMN_CELLS):
        column = np.concatenate([[0], cdf[i]]) / cdf[i, -1]
        out[:, i] = np.interp(quantiles, column, edges)
    return out


def plot(results, title, output, dpi):
    vmax = max(r["counts"].max() for r in results.values())
    norm = LogNorm(vmin=1, vmax=vmax)
    fig, axes = plt.subplots(1, len(results), figsize=(11, 5.4), layout="constrained")
    fig.patch.set_facecolor(SURFACE)

    for ax, (name, r) in zip(axes, results.items()):
        panel_title, symbol = QUANTITIES[name]
        edges, counts = r["edges"], r["counts"]
        lo, hi = edges[0], edges[-1]
        centers = 0.5 * (edges[:-1] + edges[1:])

        ax.set_facecolor(SURFACE)
        mesh = ax.pcolormesh(edges, edges, np.ma.masked_equal(counts.T, 0),
                             cmap=DENSITY, norm=norm, rasterized=True)
        ax.plot([lo, hi], [lo, hi], color=INK_SECONDARY, lw=1, ls=(0, (4, 3)), label="y = x")
        q05, q50, q95 = conditional_quantiles(counts, edges)
        ax.plot(centers, q50, color=ACCENT, lw=2, solid_capstyle="round", label="Median prediction")
        ax.plot(centers, q05, color=ACCENT, lw=1, ls=(0, (2, 2)), label="5–95% of predictions")
        ax.plot(centers, q95, color=ACCENT, lw=1, ls=(0, (2, 2)))

        m = metrics(r["sums"])
        ax.text(0.03, 0.97,
                f"Rel. L2 error  {100 * m['rel_l2']:.2f}%\n"
                f"R²  {m['r2']:.4f}\n"
                f"OLS slope  {m['slope']:.3f}",
                transform=ax.transAxes, va="top", ha="left", color=INK, fontsize=9, linespacing=1.5,
                bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=AXIS, lw=0.8, alpha=0.9))

        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(panel_title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel(f"True ${symbol}$", color=INK_SECONDARY)
        ax.set_ylabel(f"Predicted ${symbol}$", color=INK_SECONDARY)
        ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(AXIS)

    # One shared legend outside the panels, so it never covers off-diagonal cells.
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncols=3,
               fontsize=9, frameon=False, labelcolor=INK_SECONDARY)
    cbar = fig.colorbar(mesh, ax=axes, shrink=0.8, pad=0.02)
    cbar.set_label("Cells per bin", color=INK_SECONDARY)
    cbar.ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)
    cbar.outline.set_edgecolor(AXIS)
    fig.suptitle(title, color=INK, fontsize=12)

    # The outside legend needs one extra layout pass, or the suptitle overlaps panel titles.
    fig.draw_without_rendering()
    fig.savefig(output, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Density parity plots of predicted vs. true p and |WSS|.")
    parser.add_argument("eval_dir", help="Evaluation output directory with test_summary.json and predictions/")
    parser.add_argument("--output", help="Figure path; the extension sets the format (default: EVAL_DIR/parity.png)")
    parser.add_argument("--bins", type=int, default=300, help="Histogram bins per axis")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()
    if args.bins <= 0:
        parser.error("--bins must be positive")

    with open(os.path.join(args.eval_dir, "test_summary.json")) as f:
        summary = json.load(f)
    if not summary.get("predictions_dir"):
        raise SystemExit("This evaluation saved no predictions; rerun eval_ahmedml.py with --save_predictions")
    # Resolve next to the summary so copied evaluation directories still work.
    prediction_dir = os.path.join(args.eval_dir, "predictions")
    paths = [os.path.join(prediction_dir, f"{run}.npz") for run in summary["runs"]]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise SystemExit(f"Missing saved predictions for {len(missing)} runs, e.g. {missing[0]}")

    results = accumulate(paths, args.bins)

    # Guard against predictions from a different evaluation than the summary.
    for name, key in [("pressure", "global_pressure_l2re"), ("wss", "global_wss_magnitude_l2re")]:
        m = metrics(results[name]["sums"])
        if not np.isclose(m["rel_l2"], summary[key], rtol=1e-3):
            raise RuntimeError(f"{name} rel. L2 {m['rel_l2']:.6g} does not match {key} {summary[key]:.6g}")
        print(f"{name:8s}  rel. L2 {100 * m['rel_l2']:.3f}%  R² {m['r2']:.5f}  OLS slope {m['slope']:.4f}")

    cells = results["pressure"]["sums"][0]
    title = (f"{summary['model']} · {summary['inference_mode']} · "
             f"{len(paths)} test geometries · {cells / 1e6:.1f}M cells")
    output = args.output or os.path.join(args.eval_dir, "parity.png")
    plot(results, title, output, args.dpi)
    print("Saved:", output)


if __name__ == "__main__":
    main()
