import json
import sys

import pytest

from visualization import plot_accuracy_cost


RUNS = ["run_4", "run_11", "run_12"]


def write_summary(directory, model, **overrides):
    summary = {
        "model": model,
        "inference_mode": "full_geometry",
        "runs": RUNS,
        "num_test_geometries": len(RUNS),
        "evaluation_seconds": 30.0,
        "peak_gpu_memory_bytes": 6 * 2**30,
        "global_pressure_l2re": 0.05,
        "global_wss_magnitude_l2re": 0.08,
    }
    summary.update(overrides)
    directory.mkdir(parents=True)
    (directory / "test_summary.json").write_text(json.dumps(summary))
    return directory


@pytest.fixture
def eval_dirs(tmp_path):
    return [
        write_summary(tmp_path / "t3f", "transolver3_full"),
        write_summary(tmp_path / "t3e", "transolver3_efficient", peak_gpu_memory_bytes=3 * 2**30),
        write_summary(tmp_path / "lnf", "linearno_full", global_pressure_l2re=0.045),
        write_summary(tmp_path / "lne", "linearno_efficient", evaluation_seconds=12.0),
        write_summary(tmp_path / "lrf", "lrsa_full", peak_gpu_memory_bytes=20 * 2**30),
        write_summary(tmp_path / "lre", "lrsa_efficient"),
    ]


def test_costs_come_from_summary_fields(eval_dirs):
    summaries = plot_accuracy_cost.load_summaries([str(d) for d in eval_dirs])
    by_model = {s["model"]: s for s in summaries}
    assert plot_accuracy_cost.cost(by_model["transolver3_efficient"], "memory") == 3
    assert plot_accuracy_cost.cost(by_model["lrsa_full"], "memory") == 20
    assert plot_accuracy_cost.cost(by_model["linearno_efficient"], "time") == 4
    assert plot_accuracy_cost.split_model("linearno_efficient") == ("linearno", "efficient")


def test_rejects_summaries_that_are_not_comparable(tmp_path, eval_dirs):
    smoke = write_summary(tmp_path / "smoke", "lrsa_full", runs=["run_4"], num_test_geometries=1)
    with pytest.raises(ValueError, match="same test runs"):
        plot_accuracy_cost.load_summaries([str(eval_dirs[0]), str(smoke)])

    chunkwise = write_summary(tmp_path / "chunkwise", "lrsa_full", inference_mode="chunkwise")
    with pytest.raises(ValueError, match="Mixed inference modes"):
        plot_accuracy_cost.load_summaries([str(eval_dirs[0]), str(chunkwise)])

    with pytest.raises(ValueError, match="appears twice"):
        plot_accuracy_cost.load_summaries([str(eval_dirs[0]), str(eval_dirs[0] / "test_summary.json")])

    unknown = write_summary(tmp_path / "unknown", "transolver2_full")
    with pytest.raises(ValueError, match="Unknown model"):
        plot_accuracy_cost.load_summaries([str(unknown)])

    cpu = plot_accuracy_cost.load_summaries([str(write_summary(tmp_path / "cpu", "lrsa_full",
                                                               peak_gpu_memory_bytes=None))])[0]
    with pytest.raises(ValueError, match="not evaluated on a GPU"):
        plot_accuracy_cost.cost(cpu, "memory")


@pytest.mark.parametrize("cost", ["memory", "time"])
def test_main_writes_figure_for_all_or_some_models(eval_dirs, tmp_path, monkeypatch, cost):
    for dirs, name in [(eval_dirs, "all.png"), (eval_dirs[:3], "partial.png")]:
        output = tmp_path / f"{cost}_{name}"
        monkeypatch.setattr(sys, "argv", ["plot_accuracy_cost", *map(str, dirs),
                                          "--cost", cost, "--output", str(output)])
        plot_accuracy_cost.main()
        assert output.stat().st_size > 0
