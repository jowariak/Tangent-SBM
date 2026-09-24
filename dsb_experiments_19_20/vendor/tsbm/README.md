<h1 align='center'>Twisted Schrödinger Bridge Matching (TSBM)</h1>
<div align="center">
  <strong>Maxence Noble</strong><sup>1</sup><b>&ensp;</b>&ensp;
  <strong>Marie Scheid</strong><sup>1</sup><b>&ensp;</b>&ensp;
  <strong>Yazid Janati</strong><sup>2,3</sup><b>&ensp;</b>&ensp;
  <strong>Eric Moulines</strong><sup>3,4</sup><b>&ensp;</b>&ensp;
  <strong>Alain Durmus</strong><sup>1</sup><br>

  <sup>1</sup>Ecole polytechnique &emsp; 
  <sup>2</sup>Institute of Foundation Models  &emsp; 
  <sup>3</sup>MBZUAI &emsp; 
  <sup>4</sup>EPITA
</div>

## Framework

**Twisted Schrödinger Bridge Matching** (**TSBM**) is a diffusion-based learning algorithm for trajectory inference between two prescribed distributions under path-dependent constraints.
TSBM addresses the generalized—here referred to as *twisted*—Schrödinger Bridge problem over the time interval $[0,T]$. The formulation involves a time-dependent potential
$(t,x) \to V_t(x)$,
which is assumed to be continuously differentiable with respect to its spatial variable. We consider two settings:

* **Continuous-time setting.** The potential $V$ is additionally assumed to be continuous with respect to time. The twisted SB problem reads as:

  ![Continuous-time formulation](assets/latex/gsb_continuous.png)

* **Discrete-time setting.** The potential $V$ is defined only at a collection of prescribed times $t_k$. The twisted SB problem reads as:

  ![Discrete-time formulation](assets/latex/gsb_discrete.png)

In both settings, $\mu$ and $\nu$ denote the prescribed marginal distributions at the boundary times $t=0$ and $t=T$, respectively.

When $V\equiv 0$, the twisted formulation reduces to the classical Schrödinger Bridge problem, which is closely related to dynamic optimal transport. In this case, TSBM coincides with [**Diffusion Schrödinger Bridge Matching**](https://arxiv.org/abs/2303.16852) (**DSBM**), which is re-implemented in this codebase.

This repository builds upon the [official implementation](https://github.com/facebookresearch/generalized-schrodinger-bridge-matching/tree/main) of [**Generalized Schrödinger Bridge Matching**](https://arxiv.org/abs/2310.02233) (**GSBM**), a previous diffusion-based method designed to solve generalized Schrödinger Bridge problems. Its implementation is included in this codebase.

## Installation

```bash
conda env create -f environment.yml
conda activate bridge_matching
pip install -e .
```


## Implementation

TSBM is a faithful PyTorch implementation of the Iterative Markovian Fitting (IMF) algorithm. IMF was originally introduced by DSBM for the standard Schrödinger Bridge problem and is generalized here to the twisted setting.
Starting from samples drawn from the independent coupling between $\mu$ and $\nu$, the algorithm iteratively updates the coupling between the two boundary distributions. Under suitable assumptions, this coupling progressively converges to the solution of the twisted Schrödinger Bridge problem.

The training logic is implemented in [`bridge/trainer_tsbm.py`](bridge/trainer_tsbm.py). In [its basic form](bridge/trainer_tsbm.py#L547) , the IMF algorithm alternates between the forward and backward directions and repeatedly performs the following two steps:

1. **[Reciprocal projection](bridge/trainer_tsbm.py#L719).**
   The exact twisted bridge is generally intractable (in the standard setting with zero-potential, it is simply the Brownian bridge). We approximate it using a Gaussian stochastic interpolant obtained by solving a path-space variational problem.

   We use the lightweight [spline parameterization](bridge/spline/gaussian_path.py) introduced by GSBM rather than a neural-network parameterization. This provides an accurate Gaussian approximation of the intermediate marginals of the twisted bridge and enables efficient sampling from its multi-marginal distributions.

   The main difference between TSBM and GSBM at this stage lies in the variational objective. This optimization step is relatively inexpensive. Diagnostic plots can also be generated to assess the quality of the approximation; see the corresponding [plotting utilities](bridge/runners/plotters.py#L345).

2. **[Markovian projection](bridge/trainer_tsbm.py#L638).**
   This step learns marginal-preserving dynamics associated with the mixture of twisted bridges induced by the current coupling between the boundary distributions, and is the most computationally demanding part of the SB training procedure.

   The TSBM and GSBM objectives differ fundamentally. GSBM directly uses the approximate variational bridge from the Reciprocal step in its regression loss. In contrast, TSBM implements the exact Markovian projection loss by regressing the dynamics drift against a combination of:

   * the standard Brownian bridge drift used in DSBM; and
   * a correction term proportional to the gradient of the constraint potential, *evaluated at future states of the trajectory*.

   The two objectives are available in [`bridge/sde/diffusion_bridge.py`](bridge/sde/diffusion_bridge.py#L152). To reduce the variance introduced by evaluations of the potential gradient in the TSBM loss, we implement a control-variate strategy based on lightweight neural networks.
   This behavior only concerns TSBM and is controlled through:

   ```text
   use_control_variate={True|False}
   ```
   This is set to `True` by default. Note that the TSBM objective is slightly more computationally expensive because it requires additional trajectory-state samples to construct the regression target. 

### Bidirectional implementation

In addition to this standard alternating implementation, we provide a so-called *bidirectional* version of DSBM, GSBM, and TSBM inspired by [SBFlow](https://arxiv.org/abs/2409.09347) that operates in the canonical SB setting.
The corresponding training function is implemented [here](bridge/trainer_tsbm.py#L467) and consists of two stages:

1. **Pretraining.**
   Forward and backward marginal-preserving dynamics are learned jointly using samples from the independent coupling.

2. **Fine-tuning.**
   The algorithm alternates between short forward and backward Markovian projection updates.

Using smaller alternating updates instead of fully optimizing one direction at a time regularly enables coupling updates, resulting in more stable training and smoother convergence toward the Schrödinger Bridge solution.
The bidirectional implementation is the default one, enabled with:

```text
use_bidir=True
```

Setting `use_bidir=False` recovers the standard alternating implementation described above.

---

## Experiments

Each experiment is identified by a `SETTING_NAME`, which corresponds to a configuration file in [`conf/setting/`](conf/setting). The associated constraint potentials are implemented in [`bridge/state_cost.py`](bridge/state_cost.py).
Setting `gsb_method=tsbm` runs TSBM, while setting `gsb_method=gsbm` runs GSBM.

The visualizations below were generated using the checkpoints provided in [this repository](checkpoints). All reported experiments were trained on a single GPU, with training times of at most a few hours. Evaluation takes less than two minutes for each setting.

### Training

The general training command is:

```bash
python train.py \
    setting={SETTING_NAME} \
    gsb_method={tsbm|gsbm} \
    use_bidir={True|False} \
    resume_from_ckpt={True|False} \
    device={cuda|mps}
```

Replace each value enclosed in `{...}` with the desired option.

When a checkpoint is available, setting `resume_from_ckpt=True` restores the saved dynamics and resumes training:

* from the latest IMF iteration for the unidirectional implementation;
* from the fine-tuning stage for the bidirectional implementation.

When no checkpoint is available, `resume_from_ckpt` should be set to `False`, which is also the default value.

### Evaluation

To compute twisted SB metrics and generate the corresponding 2D visualizations, provide the checkpoint path for the drift of the SB dynamics in
`score_checkpoint` and `score_checkpoint_finetuning`, and run:

```bash
python evaluate.py \
    setting={SETTING_NAME} \
    gsb_method={tsbm|gsbm} \
    device={cuda|mps}
```

---

### Zero-valued potential

Available setting:

* `dsbm_gmm_2d` — DSBM checkpoint available.

This experiment solves the standard Schrödinger Bridge problem between two 2D Gaussian mixture distributions, which corresponds to a dynamic optimal transport problem. Because the potential is identically zero, TSBM reduces to DSBM.

The learned forward and backward dynamics are shown below. We clearly observe that stochastic trajectories are nearly straight.

<p align="center">
  <img src="assets/final/dsbm_f.gif" width="23%" />
  <img src="assets/final/dsbm_b.gif" width="23%" />
</p>

---

### Crowd navigation

These experiments are adapted from GSBM. The original potentials are Gaussianized to satisfy the spatial differentiability requirements of the twisted formulation.
We consider both the original 2D experiments and higher-dimensional variants based on RMS-normalized distances.

#### Stunnel constraint

Available settings:

* `stunnel_2d` — TSBM and GSBM checkpoints available;
* `stunnel_10d`;
* `stunnel_50d`.

The forward and backward dynamics learned by TSBM and GSBM in the 2D setting are shown below.

<p align="center">
  <img src="assets/final/stunnel_tsbm_f.gif" width="23%" />
  <img src="assets/final/stunnel_gsbm_f.gif" width="23%" />
  <img src="assets/final/stunnel_tsbm_b.gif" width="23%" />
  <img src="assets/final/stunnel_gsbm_b.gif" width="23%" />
</p>

#### Gaussian-mixture constraints

Available settings:

* `gmm_2d` — TSBM and GSBM checkpoints available;
* `gmm_10d`;
* `gmm_50d`.

The forward and backward dynamics learned by TSBM and GSBM in the 2D setting are shown below.

<p align="center">
  <img src="assets/final/gmm_tsbm_f.gif" width="23%" />
  <img src="assets/final/gmm_gsbm_f.gif" width="23%" />
  <img src="assets/final/gmm_tsbm_b.gif" width="23%" />
  <img src="assets/final/gmm_gsbm_b.gif" width="23%" />
</p>

#### V-neck constraint

Available settings:

* `vneck_2d` — TSBM and GSBM checkpoints available;
* `vneck_10d`;
* `vneck_50d`.

The forward and backward dynamics learned by TSBM and GSBM in the 2D setting are shown below.

<p align="center">
  <img src="assets/final/vneck_tsbm_f.gif" width="23%" />
  <img src="assets/final/vneck_gsbm_f.gif" width="23%" />
  <img src="assets/final/vneck_tsbm_b.gif" width="23%" />
  <img src="assets/final/vneck_gsbm_b.gif" width="23%" />
</p>

---

### Single-cell trajectory inference

We consider single-cell trajectory inference in 2, 5, and 50 dimensions using a novel twisted potential inspired by [TrajectoryNet](https://arxiv.org/abs/2002.04461).
The potential encourages particles to remain close to a sparse set of intermediate observations, consisting of `single_cell.ratio_obs`% of the samples from the intermediate marginals. The strength of this local attraction is controlled by the inverse-temperature parameter `single_cell.beta`, which is set to `100` by default.

Before running these experiments, prepare the dataset as follows:

1. Download `ebdata_v3.h5ad` from the [Mendeley dataset](https://data.mendeley.com/datasets/hhny5ff7yj/1).
2. Place the downloaded file in [`single_cell_data/`](single_cell_data).
3. Run the notebook [`single_cell_data/extract_data.ipynb`](single_cell_data/extract_data.ipynb).

Available settings (TSBM and GSBM checkpoints are available for each one using the default value `single_cell.beta=100`):

* `single_cell_2d`;
* `single_cell_5d`;
* `single_cell_50d`.

The forward dynamics learned by TSBM and GSBM in the 2D setting are shown below.

<p align="center">
  <img src="assets/final/single_cell_2d_tsbm.gif" width="35%" />
  <img src="assets/final/single_cell_2d_gsbm.gif" width="35%" />
</p>

---

### Discrete-time quadratic attraction potential

We consider a 2D discrete-time twisted Schrödinger Bridge problem with a quadratic potential that attracts trajectories toward observed samples at prescribed intermediate times.

#### Single-observation constraint

In this setting, a single observation set is available between the two boundary distributions. The corresponding ground-truth twisted SB can be computed explicitly; see [`bridge/data/utils.py`](bridge/data/utils.py) for details.

Available setting:

* `discrete_single_2d` — TSBM and GSBM checkpoints available.

The ground-truth, TSBM, and GSBM forward dynamics are shown below.

<p align="center">
  <img src="assets/final/discrete_single_gt_f.gif" width="23%" />
  <img src="assets/final/discrete_single_tsbm_f.gif" width="23%" />
  <img src="assets/final/discrete_single_gsbm_f.gif" width="23%" />
</p>

The corresponding backward dynamics are shown below.

<p align="center">
  <img src="assets/final/discrete_single_gt_b.gif" width="23%" />
  <img src="assets/final/discrete_single_tsbm_b.gif" width="23%" />
  <img src="assets/final/discrete_single_gsbm_b.gif" width="23%" />
</p>

#### Multiple-observation constraint

In this setting, two observation sets are available between the boundary distributions.

Available setting:

* `discrete_multi_2d` — TSBM and GSBM checkpoints available.

The forward and backward dynamics learned by TSBM and GSBM are shown below.

<p align="center">
  <img src="assets/final/discrete_multi_tsbm_f.gif" width="23%" />
  <img src="assets/final/discrete_multi_gsbm_f.gif" width="23%" />
  <img src="assets/final/discrete_multi_tsbm_b.gif" width="23%" />
  <img src="assets/final/discrete_multi_gsbm_b.gif" width="23%" />
</p>


## Citation
Please consider citing our paper:
```
@article{noble2026twisted,
  title={{Twisted Schr{\"o}dinger Bridge Matching}},
  author={Noble, Maxence and Scheid, Marie and Janati, Yazid and Moulines, Eric and Durmus, Alain},
  journal = {arXiv preprint arXiv:2607.16987},  
  year={2026}
}
```