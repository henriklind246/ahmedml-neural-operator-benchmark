import json
import sys

import numpy as np
import pytest

from visualization import plot_parity


def rel_l2(true, pred):
    return float(np.linalg.norm(pred - true) / np.linalg.norm(true))


@pytest.fixture
def eval_dir(tmp_path):
    rng = np.random.default_rng(3)
    (tmp_path / "predictions").mkdir()
    true, pred = [], []
    # A model that compresses every channel by 20% plus small noise.
    for run, n in [("run_1", 500), ("run_2", 1300)]:
        t = rng.normal(size=(n, 4)).astype(np.float32)
        p = (0.8 * t + 0.05 * rng.normal(size=(n, 4))).astype(np.float32)
        np.savez(tmp_path / "predictions" / f"{run}.npz", true=t, pred=p)
        true.append(t)
        pred.append(p)
    true, pred = np.concatenate(true).astype(np.float64), np.concatenate(pred).astype(np.float64)
    wss = np.linalg.norm(true[:, 1:], axis=1)
    wss_pred = np.linalg.norm(pred[:, 1:], axis=1)
    summary = {
        "model": "linearno_full",
        "inference_mode": "full_geometry",
        "runs": ["run_1", "run_2"],
        "predictions_dir": str(tmp_path / "predictions"),
        "global_pressure_l2re": rel_l2(true[:, 0], pred[:, 0]),
        "global_wss_magnitude_l2re": rel_l2(wss, wss_pred),
    }
    (tmp_path / "test_summary.json").write_text(json.dumps(summary))
    return tmp_path, true[:, 0], pred[:, 0]


def test_histograms_keep_every_cell_and_match_direct_metrics(eval_dir):
    path, p, p_pred = eval_dir
    results = plot_parity.accumulate(sorted((path / "predictions").glob("*.npz")), bins=40)
    pressure = results["pressure"]
    edges, counts = pressure["edges"], pressure["counts"]
    # The range covers the most extreme true and predicted cells; none are clipped.
    assert counts.sum() == results["wss"]["counts"].sum() == 1800
    assert edges[0] == min(p.min(), p_pred.min()) and edges[-1] == max(p.max(), p_pred.max())

    m = plot_parity.metrics(pressure["sums"])
    assert m["cells"] == 1800
    np.testing.assert_allclose(m["rel_l2"], rel_l2(p, p_pred))
    np.testing.assert_allclose(m["r2"], 1 - ((p_pred - p) ** 2).sum() / ((p - p.mean()) ** 2).sum())
    np.testing.assert_allclose(m["slope"], np.polyfit(p, p_pred, 1)[0])

    # The column medians recover the 0.8 compression, not y = x.
    centers = 0.5 * (edges[:-1] + edges[1:])
    median = plot_parity.conditional_quantiles(counts, edges)[1]
    dense = ~np.isnan(median)
    assert dense.sum() >= 5
    np.testing.assert_allclose(median[dense], 0.8 * centers[dense], atol=edges[1] - edges[0])


def test_main_writes_figure_and_rejects_mismatched_summary(eval_dir, monkeypatch):
    path = eval_dir[0]
    monkeypatch.setattr(sys, "argv", ["plot_parity", str(path), "--bins", "40"])
    plot_parity.main()
    assert (path / "parity.png").stat().st_size > 0

    summary = json.loads((path / "test_summary.json").read_text())
    summary["global_pressure_l2re"] *= 1.1
    (path / "test_summary.json").write_text(json.dumps(summary))
    with pytest.raises(RuntimeError, match="does not match"):
        plot_parity.main()
