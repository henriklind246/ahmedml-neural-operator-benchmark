"""Accuracy-versus-cost scatter of the benchmark's model variants.

Reads the test_summary.json that `evaluation/eval_ahmedml.py` writes for each
model and plots global relative L2 error against peak GPU memory (or evaluation
time per geometry), with one panel for pressure and one for |WSS|. Each family
has one color; full models are filled, efficient models hollow, and an arrow
runs from full to efficient. Read it as:
- further down and left is better;
- a long, mostly horizontal arrow: the efficient model saves cost almost for free;
- a steep upward arrow: halving the model costs accuracy.

It needs only NumPy and Matplotlib, and refuses summaries that are not directly
comparable (different test runs or inference modes).
"""

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# Family order fixes each family's color, whichever subset of models is plotted.
FAMILIES = {
    "transolver3": ("Transolver-3", "#2a78d6"),
    "linearno": ("LinearNO", "#eb6834"),
    "lrsa": ("LRSA", "#1baf7a"),
}
SIZES = ("full", "efficient")

COSTS = {
    "memory": "Peak GPU memory (GiB)",
    "time": "Evaluation time per geometry (s)",
}

ERRORS = {
    "pressure": ("Surface pressure", "global_pressure_l2re"),
    "wss": ("Wall shear stress magnitude", "global_wss_magnitude_l2re"),
}

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def split_model(name):
    """Return (family, size) for a model argument such as 'linearno_efficient'."""
    family, _, size = name.rpartition("_")
    if family not in FAMILIES or size not in SIZES:
        raise ValueError(f"Unknown model {name!r}; expected <{'|'.join(FAMILIES)}>_<{'|'.join(SIZES)}>")
    return family, size


def load_summaries(paths):
    """Load one test_summary.json per path (a file or its evaluation directory) and check comparability."""
    summaries = []
    for path in paths:
        if os.path.isdir(path):
            path = os.path.join(path, "test_summary.json")
        with open(path) as f:
            summary = json.load(f)
        summary["source"] = path
        summaries.append(summary)
    if not summaries:
        raise ValueError("No summaries given")

    seen = {}
    for s in summaries:
        split_model(s["model"])
        if s["model"] in seen:
            raise ValueError(f"{s['model']} appears twice: {seen[s['model']]} and {s['source']}")
        seen[s["model"]] = s["source"]

    first = summaries[0]
    for s in summaries[1:]:
        if s["inference_mode"] != first["inference_mode"]:
            raise ValueError(f"Mixed inference modes: {first['model']} is {first['inference_mode']}, "
                             f"{s['model']} is {s['inference_mode']}")
        # A smoke test on a few runs is not comparable with the full test split.
        if sorted(s["runs"]) != sorted(first["runs"]):
            raise ValueError(f"{s['model']} was evaluated on {len(s['runs'])} runs, "
                             f"{first['model']} on {len(first['runs'])}; use the same test runs for every model")
    return summaries


def cost(summary, kind):
    if kind == "memory":
        peak = summary["peak_gpu_memory_bytes"]
        if peak is None:
            raise ValueError(f"{summary['model']} was not evaluated on a GPU, so it has no peak GPU memory")
        return peak / 2**30
    # Includes data loading and scoring, not only the forward pass.
    return summary["evaluation_seconds"] / summary["num_test_geometries"]


def plot(summaries, cost_kind, output, dpi):
    points = {}
    for s in summaries:
        family, size = split_model(s["model"])
        points.setdefault(family, {})[size] = s
    families = [f for f in FAMILIES if f in points]

    fig, axes = plt.subplots(1, len(ERRORS), figsize=(10.5, 4.6), layout="constrained", sharex=True)
    fig.patch.set_facecolor(SURFACE)
    x_max = max(cost(s, cost_kind) for s in summaries)

    for ax, (panel_title, key) in zip(axes, ERRORS.values()):
        ax.set_facecolor(SURFACE)
        for family in families:
            label, color = FAMILIES[family]
            xy = {size: (cost(s, cost_kind), 100 * s[key]) for size, s in points[family].items()}
            if len(xy) == 2:
                ax.annotate("", xy=xy["efficient"], xytext=xy["full"], zorder=2,
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.5, mutation_scale=11,
                                            shrinkA=7, shrinkB=7, alpha=0.7))
            for size, (x, y) in xy.items():
                filled = size == "full"
                # Full: filled with a surface ring; efficient: hollow ring in the family color.
                # The ring sits inside the filled marker's edge, so it is drawn larger to look the same size.
                ax.scatter(x, y, s=104 if filled else 64, zorder=3, linewidths=2 if filled else 2.2,
                           facecolors=color if filled else SURFACE,
                           edgecolors=SURFACE if filled else color)
            # Name the family once, beside its full model (or whichever variant is present).
            x, y = xy.get("full", next(iter(xy.values())))
            ax.annotate(label, (x, y), xytext=(8, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8.5, color=INK_SECONDARY, zorder=4)

        ax.margins(x=0.12, y=0.18)
        ax.set_xlim(0, 1.15 * x_max)
        ax.set_title(panel_title, color=INK, fontsize=11, loc="left")
        ax.set_xlabel(COSTS[cost_kind], color=INK_SECONDARY)
        ax.set_ylabel("Relative L2 error (%)", color=INK_SECONDARY)
        ax.text(0.015, 0.025, "↙ better", transform=ax.transAxes, color=MUTED, fontsize=8)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(colors=MUTED, labelcolor=INK_SECONDARY, labelsize=8)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        for spine in ("bottom", "left"):
            ax.spines[spine].set_color(AXIS)

    handles = [Line2D([], [], ls="", marker="o", ms=8, mfc=FAMILIES[f][1], mec=FAMILIES[f][1], label=FAMILIES[f][0])
               for f in families]
    handles += [Line2D([], [], ls="", marker="o", ms=8, mfc=MUTED, mec=MUTED, label="Full model"),
                Line2D([], [], ls="", marker="o", ms=8, mfc=SURFACE, mec=MUTED, mew=2, label="Efficient model")]
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles),
               fontsize=9, frameon=False, labelcolor=INK_SECONDARY)

    first = summaries[0]
    fig.suptitle(f"Accuracy vs. cost · {first['inference_mode']} · {len(first['runs'])} test geometries",
                 color=INK, fontsize=12)
    # The outside legend needs one extra layout pass, or the suptitle overlaps panel titles.
    fig.draw_without_rendering()
    fig.savefig(output, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Relative L2 error vs. GPU memory or time, one point per model.")
    parser.add_argument("summaries", nargs="+",
                        help="Evaluation directories (or their test_summary.json files), one per model")
    parser.add_argument("--cost", choices=COSTS, default="memory", help="Quantity on the x axis")
    parser.add_argument("--output", default="accuracy_cost.png",
                        help="Figure path; the extension sets the format (default: ./accuracy_cost.png)")
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    summaries = load_summaries(args.summaries)
    order = {(f, size): i for i, (f, size) in enumerate((f, size) for f in FAMILIES for size in SIZES)}
    summaries.sort(key=lambda s: order[split_model(s["model"])])

    # The same numbers as a table, so the figure is never the only record.
    print(f"{'model':24s}{'peak GiB':>10s}{'s/geom':>9s}{'p L2 %':>9s}{'|WSS| L2 %':>12s}")
    for s in summaries:
        peak = s["peak_gpu_memory_bytes"]
        peak = f"{peak / 2**30:10.2f}" if peak is not None else f"{'—':>10s}"
        print(f"{s['model']:24s}{peak}{cost(s, 'time'):9.2f}"
              f"{100 * s['global_pressure_l2re']:9.3f}{100 * s['global_wss_magnitude_l2re']:12.3f}")

    plot(summaries, args.cost, args.output, args.dpi)
    print("Saved:", args.output)


if __name__ == "__main__":
    main()
