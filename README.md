# AhmedML Neural Operator Benchmark

Benchmarking neural operator architectures on the AhmedML surface CFD dataset.

This repository compares how different neural operator designs compress, process,
and reconstruct large unstructured surface fields under a shared AhmedML
training and evaluation protocol.

## Models

The current benchmark includes:

- **Transolver-3**
- **LinearNO**
- **Efficient LinearNO**
- **LRSA**
- **Efficient LRSA**

The full models contain approximately 7.6M parameters. The efficient LinearNO
and LRSA variants contain approximately 3.85M and 4.03M parameters,
respectively.

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
├── slurm/                # Slurm scripts used for training
├── training/             # Shared training code and launchers
└── visualization/        # ParaView-compatible qualitative outputs
