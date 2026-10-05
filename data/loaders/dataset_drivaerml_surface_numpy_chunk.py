import os
import json
import numpy as np
import torch
import torch.utils.data as data
import random
import pickle
import re


class DrivAerChunkDataset(data.Dataset):
    def __init__(self, root, data_list, train=True, label_fields=None, chunk_suffix="part", norm_stats_file=None):
        """
        root: chunked data root directory, e.g. /data/DrivAerML_surface_chunks
        data_list: dict with 'train_list' and 'test_list' keys (loaded from JSON split file)
        label_fields: list of field names to load as labels; defaults to ['pMeanTrim', 'wallShearStressMeanTrim']
        chunk_suffix: prefix for chunk filename stems (e.g. "part" -> "boundary_001_points_part0.npy")
        norm_stats_file: optional path to a .pkl file with 'label_mean' and 'label_std' arrays;
                         if None, uses pre-computed statistics from the DrivAerML training set
        """
        self.root = root
        self.label_fields = label_fields if label_fields is not None else ['pMeanTrim', 'wallShearStressMeanTrim']
        self.run_list = data_list["train_list"] if train else data_list["test_list"]
        self.chunk_suffix = chunk_suffix
        self.train = train
        self.run_chunks = {}

        # Normalization statistics computed on the DrivAerML training set
        if norm_stats_file is not None:
            with open(norm_stats_file, 'rb') as f:
                stats = pickle.load(f)
            self.label_mean = np.array(stats['label_mean'])
            self.label_std = np.array(stats['label_std'])
        else:
            self.label_mean = np.array([-2.30207226e+02, -1.20971349e+00, 1.44910027e-03, -7.12132631e-02])
            self.label_std = np.array([2.68560778e+02, 2.07625744e+00, 1.35203571e+00, 1.10551982e+00])

        for run_name in self.run_list:
            run_dir = os.path.join(root, run_name)
            boundary_name = run_name.replace('run', 'boundary')
            pattern = re.compile(rf"{re.escape(boundary_name)}_points_{re.escape(chunk_suffix)}(\d+)\.npy")
            point_chunk_files = sorted([
                f for f in os.listdir(run_dir)
                if pattern.fullmatch(f)
            ], key=lambda f: int(pattern.fullmatch(f).group(1)))
            if len(point_chunk_files) == 0:
                raise ValueError(f"No chunk files found in: {run_dir}")

            parts = []
            if self.train:
                for f_name in point_chunk_files:
                    prefix = f"{boundary_name}_points_"
                    assert f_name.startswith(prefix), f"Unexpected filename format: {f_name}"
                    stem = f_name[len(prefix):-4]
                    parts.append(stem)
                self.run_chunks[run_name] = parts
            else:
                for f_name in point_chunk_files:
                    prefix = f"{boundary_name}_points_"
                    assert f_name.startswith(prefix), f"Unexpected filename format: {f_name}"
                    stem = f_name[len(prefix):-4]
                    parts.append(stem)
                self.run_chunks[run_name] = parts

        # Each geometry may have a different number of stored parts.
        self.eval_chunks = [
            (run_name, part)
            for run_name in self.run_list
            for part in self.run_chunks[run_name]
        ]
        print(f"ChunkDataset initialized with {len(self.run_list)} runs")

    def __len__(self):
        if self.train:
            return len(self.run_list)
        else:
            return len(self.eval_chunks)

    def __getitem__(self, idx):
        if self.train:
            run_name = self.run_list[idx]
            boundary_name = run_name.replace('run', 'boundary')
            parts = self.run_chunks[run_name]
            run_dir = os.path.join(self.root, run_name)

            # AhmedML / Transolver-3 amortized training subset.
            # Randomly load stride chunks until at least 100k cells are
            # available, then sample exactly 100k cells without replacement.
            target_points = 100_000

            selected_parts = list(parts)
            random.shuffle(selected_parts)

            points_parts = []
            normals_parts = []
            labels_parts = []
            total_points = 0

            for part_id in selected_parts:
                points_part = np.load(
                    os.path.join(run_dir, f"{boundary_name}_points_{part_id}.npy")
                )
                normals_part = np.load(
                    os.path.join(run_dir, f"{boundary_name}_normals_{part_id}.npy")
                )

                field_arrays = []
                for field in self.label_fields:
                    arr = np.load(
                        os.path.join(run_dir, f"{boundary_name}_{field}_{part_id}.npy")
                    )
                    if len(arr.shape) == 1:
                        arr = arr[:, None]
                    field_arrays.append(arr)

                labels_part = np.concatenate(field_arrays, axis=1)

                points_parts.append(points_part)
                normals_parts.append(normals_part)
                labels_parts.append(labels_part)

                total_points += points_part.shape[0]
                if total_points >= target_points:
                    break

            points = np.concatenate(points_parts, axis=0)
            normals = np.concatenate(normals_parts, axis=0)
            labels = np.concatenate(labels_parts, axis=0)

            if points.shape[0] < target_points:
                raise RuntimeError(
                    f"{run_name} has only {points.shape[0]} available cells, "
                    f"but {target_points} were requested."
                )

            sample_idx = np.random.choice(
                points.shape[0],
                size=target_points,
                replace=False,
            )

            points = points[sample_idx]
            normals = normals[sample_idx]
            labels = labels[sample_idx]

            labels = (labels - self.label_mean) / (self.label_std + 1e-10)

            points_chunk = torch.from_numpy(points).float()
            normals_chunk = torch.from_numpy(normals).float()
            labels_chunk = torch.from_numpy(labels).float()

            points_list = [torch.concat([points_chunk, normals_chunk], dim=-1)]
            labels_list = [labels_chunk]

            return points_list, labels_list, points_list, run_name
        else:
            run_name, part_id = self.eval_chunks[idx]
            boundary_name = run_name.replace('run', 'boundary')

            run_dir = os.path.join(self.root, run_name)

            points = np.load(os.path.join(run_dir, f"{boundary_name}_points_{part_id}.npy"))
            normals = np.load(os.path.join(run_dir, f"{boundary_name}_normals_{part_id}.npy"))

            label_arrays = []
            for field in self.label_fields:
                arr = np.load(os.path.join(run_dir, f"{boundary_name}_{field}_{part_id}.npy"))
                if len(arr.shape) == 1:
                    arr = arr[:, None]
                label_arrays.append(arr)
            labels = np.concatenate(label_arrays, axis=1)
            labels = (labels - self.label_mean) / (self.label_std + 1e-10)

            points = torch.from_numpy(points).float()
            normals = torch.from_numpy(normals).float()
            points = torch.concat([points, normals], dim=-1)
            labels = torch.from_numpy(labels).float()

            return points, labels, points, run_name


class DrivAerMLVTUChunkDataLoader(torch.utils.data.DataLoader):
    def __init__(self, dataset, batch_size=1, sampler=None, num_workers=8):
        super().__init__(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
        self.run_list = dataset.run_list
