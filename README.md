# AhmedML Neural Operator Benchmark

Benchmarking neural operator architectures on the AhmedML surface CFD dataset.

This repository compares how different neural operator designs compress, process,
and reconstruct large unstructured surface fields under a shared AhmedML
training and evaluation protocol.

## Models

The current benchmark includes:

- **Transolver-3**
- **Efficient Transolver-3**
- **LinearNO**
- **Efficient LinearNO**
- **LRSA**
- **Efficient LRSA**

The full models contain approximately 7.6M parameters. The efficient
Transolver-3, LinearNO, and LRSA variants contain approximately 3.87M, 3.85M,
and 4.03M parameters, respectively.

## Dataset

The benchmark uses the AhmedML surface dataset with:

- 400 training geometries
- 50 validation geometries
- 50 held-out test geometries
- approximately 1.1 million surface cells per geometry

The prediction targets are:

- mean surface pressure
- mean wall shear stress vector

Dataset files are not included in this repository.

## Repository Structure

```text
ahmedml-neural-operator-benchmark/
├── data/
│   ├── loaders/          # AhmedML dataset loading and sampling
│   └── splits/           # Fixed train/validation/test splits
├── docs/                 # Upstream implementation provenance
├── evaluation/           # Shared held-out test evaluator
├── models/               # Benchmark model implementations/wrappers
├── preprocessing/        # AhmedML surface chunking
├── slurm/                # Training and full-geometry evaluation jobs
├── training/             # Shared training code and launchers
└── visualization/        # ParaView-compatible qualitative outputs
```

## Full-geometry evaluation on MSI

All six variants can evaluate a complete geometry with shared global context
using `evaluation/eval_ahmedml.py --inference_mode full_geometry`. Existing
checkpoints are supported. From this repository on MSI, activate your training
environment and submit the six-model job array:

```bash
export AHMEDML_ROOT="/scratch.global/$USER/ahmedml"
export LRSA_ROOT="$HOME/repos/LRSA-Operator"
export PYTHON_BIN="$(command -v python)"
sbatch slurm/eval_full_geometry.slurm
```

See [full-geometry inference](docs/full_geometry_inference.md) for a recommended
smoke test, checkpoint paths, single-model jobs, output metrics, and memory
requirements. Transolver-3 and LinearNO share attention statistics across compute
chunks; LRSA processes the concatenated full cloud. The evaluator's historical
`chunkwise` mode remains available for comparison.
