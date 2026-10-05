"""Geometry-aware inference using the existing checkpoint-compatible models."""

from itertools import groupby

import torch


def _run_name(batch):
    names = batch[3]
    if isinstance(names, (tuple, list)):
        if len(names) != 1:
            raise ValueError("Evaluation requires batch_size=1")
        return names[0]
    return names


def iter_predictions(model, loader, device, mode="full_geometry", chunk_size=50000):
    """Yield (prediction, label, run) on CPU, preserving stored part order.

    The loader must be sequential with batch_size=1. Full geometry mode retains
    one geometry at a time and calls the model once with all its feature chunks.
    Splitting a stored part limits Transolver/LinearNO attention workspace without
    removing any cells or changing the global attention domain. LRSA concatenates
    these pieces internally, so chunk_size does not bound its GPU memory usage.
    """
    if mode not in {"full_geometry", "chunkwise"}:
        raise ValueError(f"Unknown inference mode: {mode}")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    model.eval()
    seen = set()
    with torch.inference_mode():
        for run_name, batches in groupby(loader, key=_run_name):
            if run_name in seen:
                raise ValueError(f"Non-contiguous geometry {run_name}; use a sequential loader")
            seen.add(run_name)
            if mode == "chunkwise":
                # Preserve the historical context: one stored part per call.
                for x, y, _pos, _name in batches:
                    pred = model([x.to(device)], use_checkpoint=False)[0]
                    yield pred.cpu(), y.cpu(), run_name
                continue

            features, labels = [], []
            for x, y, _pos, _name in batches:
                if x.shape[:2] != y.shape[:2] or x.shape[0] != 1 or x.shape[1] == 0:
                    raise ValueError(f"Invalid feature/label shapes for {run_name}")
                features.extend(x.split(chunk_size, dim=1))
                labels.extend(y.split(chunk_size, dim=1))

            try:
                inputs = [x.to(device) for x in features]
                outputs = model(inputs, use_checkpoint=False)
            except torch.cuda.OutOfMemoryError as exc:
                raise RuntimeError(
                    f"Full-geometry inference ran out of GPU memory for {run_name}. "
                    "Use a GPU with more memory. For Transolver-3/LinearNO, also try "
                    "a smaller --chunk_size; LRSA concatenates the entire geometry. "
                    "The evaluator will not fall back to independent chunks."
                ) from exc
            del inputs, features
            if len(outputs) != len(labels):
                raise RuntimeError(f"Prediction chunk count mismatch for {run_name}")
            for index, label in enumerate(labels):
                prediction = outputs[index]
                if prediction.shape != label.shape:
                    raise RuntimeError(f"Prediction/label shape mismatch for {run_name}")
                # Release each GPU output as soon as it is copied for scoring.
                outputs[index] = None
                yield prediction.cpu(), label.cpu(), run_name
            del outputs, labels, prediction
