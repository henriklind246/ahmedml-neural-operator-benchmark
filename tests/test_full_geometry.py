import csv
import io
import json
import pickle
import sys

import numpy as np
import pytest
import torch
from torch.utils.data import SequentialSampler

from data.loaders.dataset_drivaerml_surface_numpy_chunk import (
    DrivAerChunkDataset, DrivAerMLVTUChunkDataLoader,
)
from evaluation import eval_ahmedml
from evaluation.inference import iter_predictions


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(17)
    torch.set_num_threads(1)


@pytest.fixture
def surface_data(tmp_path):
    split = {"train_list": [], "test_list": ["run_1", "run_2"]}
    # Unequal part counts and part10 exercise flattening and numeric ordering.
    for run, count in [("run_1", 2), ("run_2", 11)]:
        folder = tmp_path / run
        folder.mkdir()
        for part in range(count):
            n = 3 + part % 2
            prefix = folder / f"{run.replace('run', 'boundary')}"
            np.save(f"{prefix}_points_part{part}.npy", np.full((n, 3), part + 1, dtype=np.float32))
            np.save(f"{prefix}_normals_part{part}.npy", np.ones((n, 3), dtype=np.float32))
            np.save(f"{prefix}_pMean_part{part}.npy", np.zeros(n, dtype=np.float32))
            np.save(f"{prefix}_wallShearStressMean_part{part}.npy", np.zeros((n, 3), dtype=np.float32))
    norm = tmp_path / "norm.pkl"
    norm.write_bytes(pickle.dumps({"label_mean": np.zeros(4), "label_std": np.ones(4)}))
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    dataset = DrivAerChunkDataset(str(tmp_path), split, train=False,
                                 label_fields=["pMean", "wallShearStressMean"], norm_stats_file=norm)
    loader = DrivAerMLVTUChunkDataLoader(dataset, sampler=SequentialSampler(dataset), num_workers=0)
    return dataset, loader, norm, split_path


class GlobalMeanModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.context_sizes = []

    def forward(self, chunks, use_checkpoint=True):
        assert not use_checkpoint
        assert not torch.is_grad_enabled()
        cloud = torch.cat(chunks, dim=1)
        self.context_sizes.append(cloud.shape[1])
        mean = cloud[..., :1].mean(dim=1, keepdim=True)
        return [mean.expand(1, x.shape[1], 4) for x in chunks]


def test_full_geometry_coverage_context_and_unequal_parts(surface_data):
    dataset, loader, _, _ = surface_data
    assert len(dataset) == 13
    assert dataset.eval_chunks[-1] == ("run_2", "part10")
    model = GlobalMeanModel()
    results = list(iter_predictions(model, loader, "cpu", chunk_size=2))
    assert model.context_sizes == [7, 38]
    for name in dataset.run_list:
        cloud = torch.cat([dataset[i][0] for i in range(len(dataset)) if dataset[i][3] == name])
        predictions = torch.cat([p.squeeze(0) for p, _, run in results if run == name])
        assert len(predictions) == len(cloud)
        torch.testing.assert_close(predictions, cloud[:, :1].mean().expand(len(cloud), 4))


def test_chunkwise_preserves_stored_part_context(surface_data):
    dataset, loader, _, _ = surface_data
    model = GlobalMeanModel()
    results = list(iter_predictions(model, loader, "cpu", mode="chunkwise", chunk_size=1))
    assert len(results) == len(dataset)
    assert model.context_sizes == [dataset[i][0].shape[0] for i in range(len(dataset))]


@pytest.mark.parametrize("name", sorted(eval_ahmedml.MODEL_CONFIGS))
def test_all_variants_full_geometry_matches_single_cloud_and_checkpoint(name):
    if name.startswith("lrsa"):
        pytest.importorskip("perceiverforpde")
    model = eval_ahmedml.build_model(name).eval()
    assert sum(p.numel() for p in model.parameters()) == eval_ahmedml.EXPECTED_PARAMS[name]
    # A trained checkpoint has nonzero biases; exercise these as well.
    with torch.no_grad():
        for param_name, param in model.named_parameters():
            if param_name.endswith("bias"):
                param.uniform_(-0.03, 0.03)
    x = torch.randn(1, 19, 6)
    labels = torch.randn(1, 19, 4)
    loader = [(x[:, :7], labels[:, :7], None, ["run_1"]),
              (x[:, 7:], labels[:, 7:], None, ["run_1"])]
    with torch.inference_mode():
        expected = model([x], use_checkpoint=False)[0]
    # Small compute chunks must still use the full 19-cell attention domain.
    results = list(iter_predictions(model, loader, "cpu", chunk_size=3))
    torch.testing.assert_close(torch.cat([p for p, _, _ in results], dim=1), expected, atol=2e-5, rtol=2e-4)
    torch.testing.assert_close(torch.cat([y for _, y, _ in results], dim=1), labels)
    checkpoint = io.BytesIO()
    torch.save({"model_state_dict": {"module." + k: v for k, v in model.state_dict().items()}}, checkpoint)
    checkpoint.seek(0)
    restored = eval_ahmedml.build_model(name).eval()
    restored.load_state_dict(eval_ahmedml.load_state_dict(checkpoint, "cpu"), strict=True)
    with torch.inference_mode():
        torch.testing.assert_close(restored([x], use_checkpoint=False)[0], expected)


def test_evaluator_writes_mode_coverage_and_finite_metrics(surface_data, tmp_path, monkeypatch):
    _, _, norm, split_path = surface_data
    out = tmp_path / "results"
    model = GlobalMeanModel()
    checkpoint = tmp_path / "model.pth"
    torch.save(model.state_dict(), checkpoint)
    monkeypatch.setattr(eval_ahmedml, "build_model", lambda _: model)
    monkeypatch.setitem(eval_ahmedml.EXPECTED_PARAMS, "transolver3_full", 0)
    monkeypatch.setattr(sys, "argv", ["eval", "--model", "transolver3_full",
        "--data_dir", str(tmp_path), "--json_file", str(split_path),
        "--norm_stats_file", str(norm), "--model_ckpt", str(checkpoint),
        "--out_dir", str(out), "--device", "cpu", "--num_workers", "0",
        "--inference_mode", "full_geometry", "--chunk_size", "2", "--save_predictions"])
    eval_ahmedml.main()
    summary = json.loads((out / "test_summary.json").read_text())
    assert summary["inference_mode"] == "full_geometry"
    assert summary["num_test_geometries"] == 2
    assert summary["total_test_cells"] == 45
    assert np.isfinite(summary["global_pressure_l2re"])
    rows = list(csv.DictReader((out / "test_per_case.csv").open()))
    assert [int(row["num_cells"]) for row in rows] == [7, 38]
    # Independently compute cell-weighted normalized MSE from the two clouds.
    mean1 = (3 * 1 + 4 * 2) / 7
    mean2 = sum((3 + i % 2) * (i + 1) for i in range(11)) / 38
    expected_mse = (7 * mean1 ** 2 + 38 * mean2 ** 2) / 45
    np.testing.assert_allclose(summary["global_normalized_mse"], [expected_mse] * 4, rtol=1e-6)
    # Saved fields cover every cell of each geometry, across many compute chunks.
    assert summary["predictions_dir"] == str(out / "predictions")
    for run, count, value in [("run_1", 7, mean1), ("run_2", 38, mean2)]:
        with np.load(out / "predictions" / f"{run}.npz") as saved:
            np.testing.assert_array_equal(saved["true"], np.zeros((count, 4), dtype=np.float32))
            np.testing.assert_allclose(saved["pred"], np.full((count, 4), value), rtol=1e-6)


def test_rejects_noncontiguous_geometries():
    x, y = torch.ones(1, 2, 6), torch.ones(1, 2, 4)
    batches = [(x, y, None, [name]) for name in ["run_1", "run_2", "run_1"]]
    with pytest.raises(ValueError, match="Non-contiguous"):
        list(iter_predictions(GlobalMeanModel(), batches, "cpu"))
