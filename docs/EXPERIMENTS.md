# Experiment entry points

These are preserved research entry points, not an assertion that default
arguments reproduce every retained paper configuration. Use the data metadata,
saved checkpoint configuration, and paper protocol to select each run. Training
and data generation can be expensive; inspect `--help` before starting.

| System | Data generation | Conditional baseline | Tangent training |
|---|---|---|---|
| Gaussian | `gaussian_nonlinear_data.py`, `gaussian_response_collocation.py` | `gaussian_conditional_dsbm_nonlinear.py` | `gaussian_tangent_sbm_workshop_ablation.py` |
| Double well | `double_well_data.py` | `double_well_conditional_dsbm.py` | `double_well_tangent_sbm_v2.py` |
| Reaction–diffusion | `pdebench_reaction_diffusion_official.py` | `pdebench_reaction_diffusion_conditional_dsbm.py` | `pdebench_reaction_diffusion_tangent_sbm_v3.py` |
| SNS | `spdebench_sns_response_dataset.py` | `spdebench_sns_conditional_dsbm_v2.py` | `spdebench_sns_tangent_sbm_multipair_memfix.py` |

The reaction–diffusion source named `pdebench_reaction_diffusion_data.py` is an
additional generator, not a substitute for identifying the official configuration
used in the reported runs. The official generator checks its external solver
source; consult its CLI and source provenance.

SNS uses the external solver from https://github.com/DeepIntoStreams/SPDE_hackathon.
The dataset generator checks for the expected source layout and accepts a solver
root. The solver checkout is not bundled. The exact original solver revision and
dataset metadata must accompany a complete numerical reproduction; do not
silently substitute the latest revision.

## Additional methods

| Comparison | Entry point or package |
|---|---|
| Gaussian GSBM / TSBM | `dsb_experiments_19_20/run_gaussian.py` |
| Double-well GSBM / TSBM | `double_well_conditional_gsbm.py`, `double_well_conditional_tsbm.py` |
| Double-well single-rollout control | `double_well_tangent_naive_single_rollout.py` |
| PDE GSBM | `pdebench_gsbm_official/run_pde_gsbm.py` |
| PDE TSBM / SNS GSBM | `dsb_experiments_11_12/launch.py` |
| SNS TSBM / endpoint-budget control | `dsb_experiments_13_14/` |
| SNS local-budget / intervention decision experiment | `dsb_experiments_15_16/` |
| SNS qualitative / unconditional baseline | `dsb_experiments_17_18/` |
| SNS response evaluation | `sns_final2_mc1024/evaluate.py` |
| Low-dimensional all-metric evaluation | `lowdim_mc_checks/evaluate.py` |

For the decision experiment, calibration is part of the evaluated policy. Retain
the separation between development contexts and test contexts. Its results
should not be described as uncalibrated intervention selection.

Other root-level scripts implement target shuffling/sign reversal, response
coverage, endpoint-constrained supervision, and direct/operator/flow predictors.
Package names retain the original experiment IDs for compatibility, not to imply
that every version or diagnostic is a distinct paper contribution.

## Example: double-well conditional and tangent training

After creating and validating the original double-well data under
`runs/double_well_data`, train each seed (32, 42, 52):

```bash
python double_well_conditional_dsbm.py --data-dir runs/double_well_data --run-root runs/double_well_conditional --seed 32 --total-imf 7
python double_well_tangent_sbm_v2.py --data-dir runs/double_well_data --baseline-run-root runs/double_well_conditional --run-root runs/double_well_tangent_lam0p25 --anchor-response runs/double_well_data/anchor_response.pt --response-collocation runs/double_well_data/response_collocation.pt --seed 32 --fork-imf 3 --total-imf 7 --lambda-sens 0.25 --sens-every 1 --sens-batch-size 256 --anchor-fraction 0.2
```

Iteration counts above refer to outer bridge iterations. Each iteration contains
many optimization updates; they are not counts of individual gradient steps.
