"""Pick one jointly representative and one jointly difficult test geometry.

Reads test_per_case.csv from two or more evaluation directories (for example
linearno_full and lrsa_full) and ranks every run within each model by its
per-case error (default: WSS-magnitude relative L2). Percentile ranks, rather
than raw errors, make models with different overall error levels comparable.
For a target percentile t, each run scores

    sum over models |p_m - t|  +  weight * (max_m p_m - min_m p_m)

and the lowest score wins: every model is near t, and no single model is the
outlier. t = 0.5 gives the representative case. The difficult case defaults to
t = 0.85, a hard but not extreme case: the single worst run is often an outlier
that looks unlike the rest and invites the charge of cherry-picking.

Needs only the Python standard library and NumPy, so it runs on a login node.
"""

import argparse
import csv
import json
import os

import numpy as np


METRICS = ("wss_magnitude_l2re", "pressure_l2re", "wss_vector_l2re")
SHOW_CANDIDATES = 5


def load_errors(eval_dirs, metric):
    """Return (models, runs, errors[run, model]) from each directory's per-case CSV."""
    models, per_model = [], []
    for directory in eval_dirs:
        with open(os.path.join(directory, "test_summary.json")) as f:
            models.append(json.load(f)["model"])
        with open(os.path.join(directory, "test_per_case.csv"), newline="") as f:
            per_model.append({row["run"]: float(row[metric]) for row in csv.DictReader(f)})

    if len(set(models)) != len(models):
        raise ValueError(f"Each model must appear once, got {models}")
    runs = sorted(per_model[0])
    for model, errors in zip(models[1:], per_model[1:]):
        if sorted(errors) != runs:
            raise ValueError(f"{model} was evaluated on different runs than {models[0]}; "
                             "use full test-split evaluations for every model")
    if len(runs) < 3:
        raise ValueError(f"Need at least 3 evaluated runs to rank, got {len(runs)}")
    return models, runs, np.array([[errors[run] for errors in per_model] for run in runs])


def percentile_ranks(errors):
    """Rank of each run within each model's errors, scaled to [0, 1] (0 = lowest error)."""
    ranks = np.argsort(np.argsort(errors, axis=0, kind="stable"), axis=0, kind="stable")
    return ranks / (errors.shape[0] - 1)


def scores(percentiles, target, weight):
    """Distance of every run from the target percentile, plus a penalty for models disagreeing."""
    spread = percentiles.max(axis=1) - percentiles.min(axis=1)
    return np.abs(percentiles - target).sum(axis=1) + weight * spread


def select(errors, targets, weight):
    """Return {case: (best run index, order of all runs by score)}, never reusing a run."""
    percentiles = percentile_ranks(errors)
    chosen, taken = {}, set()
    for case, target in targets.items():
        order = [i for i in np.argsort(scores(percentiles, target, weight), kind="stable") if i not in taken]
        chosen[case] = (order[0], order)
        taken.add(order[0])
    return chosen


def main():
    parser = argparse.ArgumentParser(description="Pick jointly representative and difficult test geometries.")
    parser.add_argument("eval_dirs", nargs="+",
                        help="Full test-split evaluation directories, one per model (with test_per_case.csv)")
    parser.add_argument("--metric", choices=METRICS, default="wss_magnitude_l2re", help="Per-case error to rank by")
    parser.add_argument("--difficult_percentile", type=float, default=0.85,
                        help="Target percentile for the difficult case (default 0.85)")
    parser.add_argument("--disagreement_weight", type=float, default=1.0,
                        help="Penalty on the spread of percentile ranks across models (lambda)")
    args = parser.parse_args()
    if not 0.5 < args.difficult_percentile <= 1:
        parser.error("--difficult_percentile must be in (0.5, 1]")

    models, runs, errors = load_errors(args.eval_dirs, args.metric)
    percentiles = percentile_ranks(errors)
    targets = {"representative": 0.5, "difficult": args.difficult_percentile}
    chosen = select(errors, targets, args.disagreement_weight)

    print(f"Ranking {len(runs)} runs by {args.metric} for {', '.join(models)}")
    for case, (best, order) in chosen.items():
        case_scores = scores(percentiles, targets[case], args.disagreement_weight)
        print(f"\n{case} (target percentile {100 * targets[case]:.0f}), best candidates:")
        print(f"  {'run':10s}" + "".join(f"{m:>26s}" for m in models) + f"{'score':>8s}")
        for i in order[:SHOW_CANDIDATES]:
            cells = "".join(f"{100 * errors[i, j]:>17.2f}%  (p{100 * percentiles[i, j]:3.0f})"
                            for j in range(len(models)))
            print(f"  {runs[i]:10s}{cells}{case_scores[i]:8.3f}")

    print("\nSelected:")
    for case, (best, _) in chosen.items():
        print(f"  {case:15s}{runs[best]}")
    print(f"\nRuns for export: {' '.join(runs[best] for best, _ in chosen.values())}")


if __name__ == "__main__":
    main()
