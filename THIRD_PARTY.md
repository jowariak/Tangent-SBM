# Third-party components

The conditional GSBM and TSBM adapters include selected original computational
modules. Their licenses and source manifests remain in each package's `vendor/`
directory. 

- GSBM: https://github.com/facebookresearch/generalized-schrodinger-bridge-matching
  (revision `a8ab5b500dcea8b1c0df84822188f69758a08442` where supplied).
- TSBM: https://github.com/maxencenoble/twisted-sb-matching
  (exact frozen revision recorded in each `vendor/tsbm/manifest.json`).
- External SNS simulator: https://github.com/DeepIntoStreams/SPDE_hackathon
  (not bundled; its own license applies).

Fetch scripts may obtain missing upstream modules. Dataset and method citations are given in the accompanying paper.
