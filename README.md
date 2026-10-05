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
├── docs/                 # Evaluation instructions and upstream provenance
├── evaluation/           # Shared held-out test evaluator
├── models/               # Benchmark model implementations/wrappers
├── preprocessing/        # AhmedML surface chunking
├── slurm/                # Training and full-geometry evaluation jobs
├── training/             # Shared training code and launchers
├── tests/                # Full-geometry inference correctness checks
└── visualization/        # ParaView outputs and prediction parity plots
```

## How the pipeline fits together

This repository trains and evaluates CFD surrogate models: they predict surface
pressure and the three wall-shear-stress components from each surface cell's
coordinates and normal vector. The stored CFD solutions provide training labels
and held-out evaluation truth. The repository does not run the CFD solver itself.

```text
AhmedML surface meshes and CFD fields (external data)
    → preprocessing/: split cells into NumPy parts
    → data/: load parts, normalize labels, apply fixed geometry splits
        → training/ + models/: train a model → saved .pth checkpoint
        → evaluation/ + models/ + checkpoint: predict held-out geometries
            → per-case CSV and aggregate JSON metrics
```

Training samples 100,000 cells per geometry. Full-geometry evaluation uses all
stored cells of each test geometry with global context. `slurm/` provides MSI job
launchers around the Python entry points. `visualization/` provides a separate
Transolver-3 full-model script for exporting predictions to VTP for ParaView;
that script currently uses independent-chunk inference. It also provides
`plot_parity.py`, which plots predicted against true pressure and |WSS| for any
evaluated model (see [parity plots](docs/full_geometry_inference.md#parity-plots)).

## External code and data on MSI

Run the shared evaluator from **this repository** for every model family:

| Model family (full and efficient) | Implementation used for evaluation | Other source checkout needed? |
| --- | --- | --- |
| Transolver-3 | `models/Transolver_chunk_opt_matrix_mul.py` | No |
| LinearNO | `models/LinearNO_chunk_opt_matrix_mul.py` | No |
| LRSA | Local wrapper importing `perceiverforpde` | LRSA-Operator source or its installed package |

For LRSA, the job adds `$LRSA_ROOT/src` to Python's import path (default:
`$HOME/repos/LRSA-Operator/src`). It imports the upstream implementation into the
same evaluation process; there is no separate LRSA job or evaluation command.
The upstream package's runtime dependencies must also be installed. All models
need the Python environment, preprocessed data, training normalization file, and
matching checkpoint; datasets and trained weights are not shipped in this repo.
See [upstream versions](docs/upstream_versions.md) for recorded revisions.

Except for the updated `slurm/linearno_full.slurm`, the older **training** Slurm
scripts still point to `~/repos/Transolver-3` and legacy `main_ahmedml_*.py`
filenames. Some training launchers also retain
`--eval 2` branches referencing missing amortized model classes. Those scripts
have not yet been fully migrated to this repository layout. The shared
`evaluation/eval_ahmedml.py` workflow below runs from this checkout and does not
use those legacy launchers.

## Example: train LinearNO full on MSI

Activate your training environment, then submit this batch job from the MSI
login node. It requests **four A100 GPUs on one node for up to eight hours**,
with 32 CPUs and 128 GB host RAM. The batch script calls the local training
launcher; no separate LinearNO or Transolver-3 checkout is required.

```bash
cd ~/repos/ahmedml-neural-operator-benchmark
export AHMEDML_ROOT="/scratch.global/$USER/ahmedml"
export PYTHON_BIN="$(command -v python)"

sbatch slurm/linearno_full.slurm
```

This trains the 16-layer LinearNO model for 500 epochs, with one sampled geometry
per GPU per step and validation every 25 epochs (also at the final epoch).
The input chunks and normalization file must already exist. Training uses the
400/50 train/validation split, leaving the held-out test geometries for evaluation.
On successful completion, the final checkpoint is
`$AHMEDML_ROOT/linearno_full_500/model_500.pth`, matching the evaluation launcher's
default path. Export `SAVE_DIR` before submission to choose a different output
directory for a separate training run.

The script starts four workers with `python -m torch.distributed.run`, using
`--nproc_per_node=4`, a learning rate of `0.001`, and `OMP_NUM_THREADS=8`.
Logs are written to `slurm-<job-id>.out` and `slurm-<job-id>.err` in the submission
directory. The eight-hour limit is the job's maximum runtime; completion of all
500 epochs within that time has not been measured.

## Full-geometry evaluation on MSI

All six variants can evaluate a complete geometry with shared global context
using `evaluation/eval_ahmedml.py --inference_mode full_geometry`. Existing
checkpoints are supported. From this repository on MSI, activate your training
environment and submit one named model:

```bash
export AHMEDML_ROOT="/scratch.global/$USER/ahmedml"
export LRSA_ROOT="$HOME/repos/LRSA-Operator"
export PYTHON_BIN="$(command -v python)"
sbatch slurm/eval_full_geometry.slurm transolver3_full
```

See [full-geometry inference](docs/full_geometry_inference.md) for a recommended
smoke test, checkpoint paths, single-model jobs, output metrics, and memory
requirements. Transolver-3 and LinearNO share attention statistics across compute
chunks; LRSA processes the concatenated full cloud. The evaluator's historical
`chunkwise` mode remains available for comparison.
