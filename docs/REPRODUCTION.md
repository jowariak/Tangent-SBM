# Reproduction status

## Included measurements

`results/gaussian_id_mc4096.json` and `results/double_well_id_mc4096.json` contain
the executed three-seed ID evaluations, their nested MC prefixes, checkpoint/data
hashes, and aggregate means/sample standard deviations. Machine-local checkpoint
paths have been removed; checkpoint hashes and configurations are preserved.
`lowdim_mc_checks/paths.json` supplies the conventional relative locations.

The Gaussian comparison includes Conditional DSBM, GSBM, TSBM and Tangent-SBM.
Double well additionally includes the single-rollout Sobolev control. These
files do not reevaluate unconditional DSBM or OOD splits, and do not contain new
covariance or well-occupancy measurements.

## Validation

The release validator checks Python syntax, imported source hashes, pinned
third-party source manifests, and aggregate statistics against the per-seed
records. The small PyTorch self-test checks known linear-drift tangents and
paired finite changes. These checks do not establish that a fresh training run
reproduces the paper end to end.

## Items not in this package

- Trained model checkpoints and generated endpoint/response datasets.
- Exact original environment lockfiles and hardware/runtime measurements.
- A pinned copy of the external SPDEBench solver and its training-data metadata.
- All paper ablation/decision/field-result logs and the current manuscript.
- The final PDE baseline validation/stopping workflow as a documented one-command run.

The authors should attach data metadata, checkpoint downloads, exact dependency
versions and the remaining experiment configurations before advertising full
end-to-end reproducibility. An independent clean-environment training run has
not been performed for this assembled release.

## Metric interpretation

More Monte Carlo samples reduce uncertainty in estimated model means; they do
not change model weights or simulator reference accuracy. Use a common declared
budget across compared models. A mean-target objective does not specify the
physical distribution of trajectory responses. Evaluation improvements on mean
metrics alone do not establish superiority over deterministic response predictors.
