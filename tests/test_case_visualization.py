import csv
import json
import sys

import numpy as np
import pytest

from visualization import export_wss_error_vtp as export
from visualization import select_cases


RUNS = [f"run_{i}" for i in range(11)]


def write_per_case(directory, model, wss_errors):
    directory.mkdir(parents=True)
    (directory / "test_summary.json").write_text(json.dumps({"model": model, "runs": list(wss_errors)}))
    with open(directory / "test_per_case.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run", "pressure_l2re", "wss_magnitude_l2re"])
        writer.writeheader()
        for run, error in wss_errors.items():
            writer.writerow({"run": run, "pressure_l2re": 0.05, "wss_magnitude_l2re": error})
    return directory


@pytest.fixture
def per_case_dirs(tmp_path):
    # LinearNO ranks run_i at percentile i/10. LRSA agrees, except it finds run_5 hard
    # (percentile 0.9), which shifts run_6..run_9 down one rank.
    lrsa_rank = {i: i for i in range(11)}
    lrsa_rank.update({5: 9, 6: 5, 7: 6, 8: 7, 9: 8})
    linearno = {f"run_{i}": 0.08 + 0.001 * i for i in range(11)}
    lrsa = {f"run_{i}": 0.10 + 0.002 * lrsa_rank[i] for i in range(11)}
    return [write_per_case(tmp_path / "linearno", "linearno_full", linearno),
            write_per_case(tmp_path / "lrsa", "lrsa_full", lrsa)]


def test_percentile_ranks_scale_each_model_to_unit_interval():
    errors = np.array([[0.3, 5.0], [0.1, 7.0], [0.2, 6.0]])
    np.testing.assert_allclose(select_cases.percentile_ranks(errors), [[1, 0], [0, 1], [0.5, 0.5]])


def test_selection_prefers_runs_where_models_agree(per_case_dirs):
    models, runs, errors = select_cases.load_errors(per_case_dirs, "wss_magnitude_l2re")
    assert models == ["linearno_full", "lrsa_full"]

    def pick(target, weight):
        best, _ = select_cases.select(errors, {"case": target}, weight)["case"]
        return runs[best]

    # run_4 sits at (0.4, 0.4); run_6 at (0.6, 0.5): closer to the median but the models disagree.
    assert pick(0.5, weight=2.0) == "run_4"
    assert pick(0.5, weight=0.5) == "run_6"
    # run_5 is LinearNO's median but an LRSA outlier, so it is never the representative case.
    assert pick(0.5, weight=0.0) != "run_5"
    # run_9 sits at (0.9, 0.8), the closest jointly high pair to the 85th percentile.
    assert pick(0.85, weight=1.0) == "run_9"


def test_selection_rejects_different_run_sets(per_case_dirs, tmp_path):
    partial = write_per_case(tmp_path / "partial", "transolver3_full", {"run_1": 0.1, "run_2": 0.2, "run_3": 0.3})
    with pytest.raises(ValueError, match="different runs"):
        select_cases.load_errors([per_case_dirs[0], partial], "wss_magnitude_l2re")


def test_select_main_reports_two_distinct_runs(per_case_dirs, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["select_cases", *map(str, per_case_dirs)])
    select_cases.main()
    out = capsys.readouterr().out
    assert "representative run_4" in " ".join(out.split())
    assert "difficult run_9" in " ".join(out.split())
    assert "Runs for export: run_4 run_9" in out


def stored(fields, num_parts):
    """Mimic preprocessing + evaluation: stride-split into parts, concatenate by part index."""
    return np.concatenate([fields[k::num_parts] for k in range(num_parts)])


def test_mesh_order_undoes_stride_parts_with_uneven_sizes():
    fields = np.arange(46, dtype=np.float64).reshape(23, 2)
    np.testing.assert_array_equal(export.to_mesh_order(stored(fields, 4), 4), fields)


def test_wss_arrays_use_magnitude_error():
    true = np.array([[1.0, 3.0, 4.0, 0.0]])
    arrays = export.wss_arrays(true, {"m": np.array([[9.0, 6.0, 0.0, 8.0]])})
    assert arrays["wss_magnitude_true"][0] == 5
    assert arrays["m_wss_magnitude_pred"][0] == 10
    assert arrays["m_wss_magnitude_abs_error"][0] == 5


@pytest.fixture
def export_case(tmp_path):
    pv = pytest.importorskip("pyvista")
    rng = np.random.default_rng(0)
    mesh = pv.Plane(i_resolution=6, j_resolution=5)
    n = mesh.n_cells
    mesh.cell_data["pMean"] = rng.normal(size=n)
    mesh.cell_data["wallShearStressMean"] = rng.normal(size=(n, 3))
    (tmp_path / "raw" / "run_3").mkdir(parents=True)
    mesh.save(tmp_path / "raw" / "run_3" / "boundary_3.vtp")

    # Four stored parts, as the evaluator's data_dir would hold them.
    (tmp_path / "chunks" / "run_3").mkdir(parents=True)
    for k in range(4):
        np.save(tmp_path / "chunks" / "run_3" / f"boundary_3_points_part{k}.npy", np.zeros((0, 3)))

    true = np.column_stack([mesh.cell_data["pMean"], mesh.cell_data["wallShearStressMean"]]).astype(np.float32)
    eval_dirs = []
    for model, factor in [("linearno_full", 1.1), ("lrsa_full", 0.8)]:
        directory = tmp_path / model
        (directory / "predictions").mkdir(parents=True)
        (directory / "test_summary.json").write_text(json.dumps(
            {"model": model, "runs": ["run_3"], "data_dir": str(tmp_path / "chunks")}))
        np.savez(directory / "predictions" / "run_3.npz", true=stored(true, 4), pred=stored(factor * true, 4))
        eval_dirs.append(str(directory))
    return pv, tmp_path, eval_dirs, mesh


def test_export_writes_errors_in_mesh_cell_order(export_case, monkeypatch):
    pv, tmp_path, eval_dirs, mesh = export_case
    monkeypatch.setattr(sys, "argv", ["export", *eval_dirs, "--runs", "run_3",
                                      "--raw_dir", str(tmp_path / "raw"), "--out_dir", str(tmp_path / "out")])
    export.main()

    out = pv.read(tmp_path / "out" / "run_3.vtp")
    assert out.n_cells == mesh.n_cells and "pMean" not in out.cell_data
    magnitude = np.linalg.norm(mesh.cell_data["wallShearStressMean"], axis=1)
    np.testing.assert_allclose(out.cell_data["wss_magnitude_true"], magnitude, rtol=1e-6)
    np.testing.assert_allclose(out.cell_data["linearno_full_wss_magnitude_abs_error"], 0.1 * magnitude, rtol=1e-5)
    np.testing.assert_allclose(out.cell_data["lrsa_full_wss_magnitude_pred"], 0.8 * magnitude, rtol=1e-5)


def test_export_rejects_wrong_ordering_and_missing_predictions(export_case, monkeypatch):
    _, tmp_path, eval_dirs, _ = export_case
    base = ["export", *eval_dirs, "--raw_dir", str(tmp_path / "raw"), "--out_dir", str(tmp_path / "out")]

    monkeypatch.setattr(sys, "argv", [*base, "--runs", "run_3", "--num_parts", "5"])
    with pytest.raises(RuntimeError, match="does not match"):
        export.main()

    monkeypatch.setattr(sys, "argv", [*base, "--runs", "run_3", "run_7"])
    with pytest.raises(SystemExit, match="--runs run_3 run_7 --save_predictions"):
        export.main()
