# Learning Stochastic Transport with Mechanistic Sensitivities

Research code for **Tangent Schrödinger Bridge Matching (Tangent-SBM)**.
The method supervises how learned stochastic transports respond to physical
parameter changes, in addition to fitting endpoint observations.

The code distinguishes realization-level squared response matching from
independent-rollout matching of conditional-mean responses. Gaussian and
PDEBench use squared response matching; double well and stochastic
Navier–Stokes (SNS) include the mean-response objective and its single-rollout
control.

## Release contents

- Dataset generation, conditional/unconditional bridges, and tangent training.
- Conditional GSBM/TSBM adapters with third-party attribution and source manifests.
- Target, coverage, rollout, budget, and alternative-predictor controls.
- Frozen-checkpoint evaluation and resumable Monte Carlo convergence checks.


## Setup

Use Python 3.11 and a PyTorch installation appropriate for your hardware.
The supplied low-dimensional MC results were produced with PyTorch 2.8.0+cu128.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tools/validate_release.py
python tools/summarize_results.py
python lowdim_mc_checks/self_test.py
```

GPU training is recommended for field experiments. The last command runs small
CPU tests of tangent propagation, endpoint prediction, and paired finite changes.
It does not train a model. SNS additionally requires the external SPDEBench
solver; see [dataset and experiment entry points](docs/EXPERIMENTS.md).

Run scripts from the repository root. Each primary script supports `--help`.
The original file names and adapter directories are retained so local imports
and checkpoint conventions remain recognizable.

## Reevaluate frozen checkpoints

Place the original data and checkpoint directories under `runs/`; edit
`lowdim_mc_checks/paths.json` if necessary.

```bash
python lowdim_mc_checks/evaluate.py --dataset gaussian --check-only
python lowdim_mc_checks/evaluate.py --dataset gaussian --mc 8 16 32 64 128 256 512 1024 2048 4096
python lowdim_mc_checks/evaluate.py --dataset double_well --mc 8 16 32 64 128 256 512 1024 2048 4096
```


## Documentation

- [Experiment and dataset entry points](docs/EXPERIMENTS.md)
- [Third-party code](THIRD_PARTY.md)

`SOURCE_MANIFEST.json` records hashes of the imported source files. Bundled
third-party components retain their original licenses. A project-wide license
for the authors' original code has not yet been selected.
