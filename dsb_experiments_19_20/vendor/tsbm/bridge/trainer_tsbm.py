import os

import time
import math
import torch
import numpy as np
import hydra
from torch.utils.data import DataLoader
from tqdm import tqdm

from .sde import *
from .data import *
from .data.utils import gather
from .models.utils import get_state_dict, DirectionalScoreWrapper
from .sde.diffusion_bridge import get_sde_timesteps
from .spline import ZeroBaseDrift, BrownianBridgeDrift
from .spline import gaussian_path as gpath_lib
from .runners import *
from .runners.config_getters import get_score_model, get_control_variate_model, get_optimizer, get_plotter, get_logger, \
    get_state_cost

DISC_STATE_COSTS = ['quadratic']  # discrete time state cost names


def _cfg_get(cfg, name):
    value = getattr(cfg, name)
    if value is None:
        raise ValueError(f"Missing required config value: {name}")
    return value


class TSBM:
    def __init__(self, init_ds, final_ds, test_init_ds, test_final_ds, shape_x, args, accelerator=None):
        self.accelerator = accelerator
        self.device = self.accelerator.device

        self.args = args

        # dataset params
        self.init_ds = init_ds
        self.final_ds = final_ds
        self.test_init_ds = test_init_ds
        self.test_final_ds = test_final_ds
        self.shape_x = shape_x
        self.len_ds = min(len(self.init_ds), len(self.final_ds))
        self.len_test_ds = min(len(self.test_init_ds), len(self.test_final_ds))

        # general SB params
        self.T = self.args.T
        self.eps = self.args.eps_train
        self.eps_test = self.args.eps_test
        self.sigma = self.args.sigma
        self.use_bidir = bool(self.args.use_bidir)

        # train/test params
        self.N = int(self.args.train_num_steps)  # (num_steps+1) = number of time-steps on (0,T)
        self.test_N = int(self.args.test_num_steps)
        self.batch_size = int(self.args.batch_size)
        self.loss_param = self.args.loss_param
        self.use_control_variate = self.args.use_control_variate
        self.resume = self.args.resume_from_checkpoint

        # unidirectional training
        self.n_imf = self.args.outer_iters
        self.starting_direction = self.args.fb_sequence[0]
        assert self.starting_direction in ['f', 'b']
        self.num_epochs = int(self.args.num_epochs)
        self.first_it_refresh_every = int(self.args.first_it_refresh_every)
        self.next_it_refresh_every = int(self.args.next_it_refresh_every)
        self.score_lr = self.args.score_lr
        self.cv_lr = self.args.cv_lr

        # bidirectional training
        self.pretrain_epochs = int(self.args.bidir.pretrain_epochs)
        self.finetune_epochs = int(self.args.bidir.finetune_epochs)
        self.pretrain_refresh_every = int(self.args.bidir.pretrain_refresh_every)
        self.finetune_refresh_every = int(self.args.bidir.finetune_refresh_every)
        self.pretrain_lr = self.args.bidir.pretrain_lr
        self.finetune_lr = self.args.bidir.finetune_lr
        self.pretrain_cv_lr = self.args.bidir.pretrain_cv_lr
        self.finetune_cv_lr = self.args.bidir.finetune_cv_lr

        # TSBM-specific
        self.use_state_cost_annealing = self.args.use_state_cost_annealing
        self.max_epoch_annealing = self.args.max_epoch_annealing
        self.factor_state_cost = self.args.factor_state_cost
        self.sample_s = True
        self.num_s_per_t = int(self.args.num_s_per_t)

        # control points for conditional sampling
        self.N_mean = None

        # Langevin sampler
        self.langevin = None

        # set time discretization for train setting
        self.sde_ts, dt_train = get_sde_timesteps(self.T, self.N, self.eps, self.device)
        assert 2 * self.eps <= dt_train, 'Decrease train_num_steps or eps_train.'

        # set time discretization fo test setting
        self.test_sde_ts, dt_test = get_sde_timesteps(self.T, self.test_N, self.eps_test, self.device)
        assert 2 * self.eps_test <= dt_test, 'Decrease test_num_steps or eps_test.'

        # build state cost
        self.build_state_cost()

        # build checkpoints
        self.ckpt_dir = self.args.checkpoint_dir
        self.build_checkpoints()

        # build models
        self.build_models()

        # build optimizers
        self.build_optimizers()

        # get logger
        self.logger = self.get_logger('train_logs')
        self.save_logger = self.get_logger('test_logs')

        # get data
        self.build_save_dataloaders()
        self.spline_batch_size = min(int(self.args.spline_batch_size) * self.accelerator.num_processes, self.len_ds)

        # get plotter
        self.plot_npar = min(int(self.args.plot_npar), self.len_test_ds)
        self.plotter = self.get_plotter()

        # saving params
        self.stride = self.len_ds * int(self.args.test_epoch_stride) // self.batch_size
        self.stride_log = self.len_ds * int(self.args.log_epoch_stride) // self.batch_size
        self.stride_ckpt = self.len_ds * int(self.args.save_epoch_stride) // self.batch_size

    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def clear(self):
        self.accelerator.free_memory()
        torch.cuda.empty_cache()

    def compute_current_step(self, i, n, num_iter):
        """Computes the total SGD step id."""
        return i + num_iter * (n - 1)

    def build_state_cost(self):
        """Builds the state cost function: either continuous-time (including zero state cost) or discrete-time"""
        self.V = get_state_cost(self.args)

        if self.args.cost_name in DISC_STATE_COSTS:
            self.disc_cost_times = self.V.y_times
            self.K = self.V.K
            self.discrete_cfg = {'cost_times': self.disc_cost_times, 'num_cost_times': self.K}
            self.accelerator.print('cfg', self.discrete_cfg)
            self.V.get_num_cost_times_per_t = self.get_num_cost_times_per_t
        else:
            self.disc_cost_times = None
            self.K = None
            self.discrete_cfg = None

    def build_control_times(self):
        """Build the times for mean control points (including boundaries 0 and T):
            - control_point_indexes: (N_mean,) : indexes associated with self.sde_ts."""
        # uniform spacing between 0 and T, taken in 'f' direction -> first index : time 0, last index : time T
        self.control_point_indexes = torch.linspace(0, self.N, self.N_mean).long()

    def get_num_cost_times_per_t(self, t, fb='f'):
        """ Returns the number of cost times strictly greater than t (fb='f') or strictly lower than t (fb='b).
        This is only used with discrete-time state cost.

        - t (B,1)
        - fb: 'f' or 'b'

        -> num: (B,1) with values in {0, ..., self.K}"""
        assert self.disc_cost_times is not None, "This function is only supported for discrete time cost function."
        if fb == 'f':
            start = torch.searchsorted(self.disc_cost_times, t.contiguous(), right=True)  # (B,1)
            num = self.K - start  # in [0, ..., K]
        else:
            end = torch.searchsorted(self.disc_cost_times, t.contiguous(), right=False)  # (B,1)
            num = end  # in [0, ..., K]
        return num

    def build_checkpoints(self):
        """Builds the checkpoints (models & optimizers) for unidirectional and bidirectional settings."""
        self.cache_dir = './cache/'
        if self.accelerator.is_main_process:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            os.makedirs(self.cache_dir, exist_ok=True)

        if not self.resume:
            self.checkpoint_it = 1
            self.checkpoint_pass = self.starting_direction
            self.checkpoint_stage = 'pretraining'
            self.step = 1
            self.accelerator.print("Starting training from scratch.")
            return

        # Resume from given checkpoints
        self.checkpoint_it = self.args.resume_at_outer_iter
        self.checkpoint_pass = self.args.resume_at_dir
        self.checkpoint_stage = self.args.resume_at_stage
        self.step = self.args.resume_at_sgd_step

        # Bidirectional setting : we separate between pretraining and finetuning
        if self.use_bidir:
            assert self.checkpoint_stage in ["pretraining", "finetuning"]

            if self.checkpoint_stage == "pretraining":
                assert self.args.score_checkpoint_pretraining is not None, 'Pretraining score ckpt is not given.'
                self.score_checkpoint_pretraining = hydra.utils.to_absolute_path(self.args.score_checkpoint_pretraining)

                if self.args.optimizer_checkpoint_pretraining is not None:
                    self.optimizer_checkpoint_pretraining = hydra.utils.to_absolute_path(
                        self.args.optimizer_checkpoint_pretraining)

                if self.use_control_variate and self.args.cv_checkpoint_pretraining is not None:
                    self.cv_checkpoint_pretraining = hydra.utils.to_absolute_path(self.args.cv_checkpoint_pretraining)

            if self.checkpoint_stage == "finetuning":
                assert self.args.score_checkpoint_finetuning is not None, 'Finetuning score ckpt is not given.'
                self.score_checkpoint_finetuning = hydra.utils.to_absolute_path(self.args.score_checkpoint_finetuning)

                if self.args.optimizer_checkpoint_finetuning is not None:
                    self.optimizer_checkpoint_finetuning = hydra.utils.to_absolute_path(
                        self.args.optimizer_checkpoint_finetuning)

                if self.use_control_variate and self.args.cv_checkpoint_finetuning is not None:
                    self.cv_checkpoint_finetuning = hydra.utils.to_absolute_path(self.args.cv_checkpoint_finetuning)

            self.accelerator.print(
                f"Resuming bidirectional training at stage {self.checkpoint_stage}, SGD step {self.step}")

        # Unidirectional setting
        else:
            assert self.args.score_checkpoint is not None, 'Score ckpt is not given.'
            self.score_checkpoint = hydra.utils.to_absolute_path(self.args.score_checkpoint)

            if self.args.optimizer_checkpoint is not None:
                self.optimizer_checkpoint = hydra.utils.to_absolute_path(self.args.optimizer_checkpoint)

            if self.use_control_variate and self.args.cv_checkpoint is not None:
                self.cv_checkpoint = hydra.utils.to_absolute_path(self.args.cv_checkpoint)

            self.accelerator.print(
                f"Resuming unidirectional training at outer iteration {self.checkpoint_it}/{self.n_imf}, "
                f"in direction {self.checkpoint_pass}, SGD step {self.step}.")

    def build_models(self):
        """Builds the models for unidirectional and bidirectional settings and loads the checkpoints."""
        score_net_f_b = get_score_model(self.args)
        cv_net_f = get_control_variate_model(self.args)
        cv_net_b = get_control_variate_model(self.args)

        # Resume from checkpoints
        if self.resume:
            if self.use_bidir and self.checkpoint_stage == "pretraining":
                score_ckpt = self.score_checkpoint_pretraining
                cv_ckpt = getattr(self, "cv_checkpoint_pretraining", None)
            elif self.use_bidir:
                score_ckpt = self.score_checkpoint_finetuning
                cv_ckpt = getattr(self, "cv_checkpoint_finetuning", None)
            else:
                # unidirectional training
                score_ckpt = self.score_checkpoint
                cv_ckpt = getattr(self, "cv_checkpoint", None)

            try:
                score_net_f_b.load_state_dict(torch.load(score_ckpt, map_location="cpu"))
            except:
                state_dict = torch.load(score_ckpt, map_location="cpu")
                score_net_f_b.load_state_dict(get_state_dict(state_dict))
            self.accelerator.print('Score ckpt loaded !')

            if self.use_control_variate and cv_ckpt is not None:
                state_dict_cv = torch.load(cv_ckpt, map_location="cpu")
                cv_net_f.load_state_dict({
                    k.removeprefix("f."): v
                    for k, v in state_dict_cv.items()
                    if k.startswith("f.")
                })
                cv_net_b.load_state_dict({
                    k.removeprefix("b."): v
                    for k, v in state_dict_cv.items()
                    if k.startswith("b.")
                })
                self.accelerator.print('Control variate ckpt loaded !')

        # Define the networks
        self.net = score_net_f_b
        if self.use_control_variate:
            self.cv_net = torch.nn.ModuleDict({"f": cv_net_f, "b": cv_net_b})
        else:
            self.cv_net = {"f": None, "b": None}

    def build_optimizers(self):
        """Builds the optimizers for unidirectional and bidirectional settings and loads the checkpoints."""
        if self.use_bidir:
            # Bidirectional setting
            cv_net = self.cv_net if self.use_control_variate else None

            if self.checkpoint_stage == 'pretraining':
                score_lr, cv_lr = self.pretrain_lr, self.pretrain_cv_lr
                opt_ckpt = getattr(self, "optimizer_checkpoint_pretraining", None)
            else:
                score_lr, cv_lr = self.finetune_lr, self.finetune_cv_lr
                opt_ckpt = getattr(self, "optimizer_checkpoint_finetuning", None)
            optimizer = get_optimizer(self.net, cv_net, score_lr, cv_lr)

            if self.resume and opt_ckpt is not None:
                optimizer.load_state_dict(torch.load(opt_ckpt, map_location="cpu"))
                self.accelerator.print('Optimizer ckpt loaded !')

            if self.use_control_variate:
                self.net, self.cv_net, self.optimizer = self.accelerator.prepare(self.net, self.cv_net, optimizer, )
            else:
                self.net, self.optimizer = self.accelerator.prepare(self.net, optimizer, )
        else:
            # Unidirectional setting
            optimizer_f = get_optimizer(self.net, self.cv_net["f"], self.score_lr, self.cv_lr)
            optimizer_b = get_optimizer(self.net, self.cv_net["b"], self.score_lr, self.cv_lr)
            opt_ckpt = getattr(self, "optimizer_checkpoint", None)

            if self.resume and opt_ckpt is not None:
                state_dict_optimizer = torch.load(opt_ckpt, map_location="cpu")
                optimizer_f.load_state_dict(state_dict_optimizer["f"])
                optimizer_b.load_state_dict(state_dict_optimizer["b"])
                self.accelerator.print('Optimizer ckpt loaded !')

            self.optimizer = {'f': optimizer_f, 'b': optimizer_b}

            if self.use_control_variate:
                self.net, self.cv_net, self.optimizer["f"], self.optimizer["b"] = self.accelerator.prepare(
                    self.net, self.cv_net, self.optimizer["f"], self.optimizer["b"])
            else:
                self.net, self.optimizer["f"], self.optimizer["b"] = self.accelerator.prepare(
                    self.net, self.optimizer["f"], self.optimizer["b"])

    def save_ckpt(self, i, n, fb='f', stage='pretraining'):
        """Saves separately the score network, the control variate network and the optimizer.

        Args:
        - i (int) : SGD iteration
        - n (int) : IMF iteration
        - fb ('f' or 'b') : training direction (unidirectional setting)
        - stage ('pretraining' or 'finetuning'): training stage (bidirectional setting)"""
        if not self.accelerator.is_main_process:
            return

        # save score network
        name_net = f'net_{stage}_{n:03}_{i:07}.ckpt' if self.use_bidir else f'net_{fb}_{n:03}_{i:07}.ckpt'
        torch.save(
            self.accelerator.unwrap_model(self.net).state_dict(),
            os.path.join(self.ckpt_dir, name_net),
        )

        # save control variate network
        if self.use_control_variate:
            name_cv = f'cv_net_{stage}_{n:03}_{i:07}.ckpt' if self.use_bidir else f'cv_net_{fb}_{n:03}_{i:07}.ckpt'
            torch.save(
                self.accelerator.unwrap_model(self.cv_net).state_dict(),
                os.path.join(self.ckpt_dir, name_cv),
            )

        # save optimizer
        if self.use_bidir:
            name_opt = f'optimizer_{stage}_{n:03}_{i:07}.ckpt'
            state_dict_opt = self.optimizer.state_dict()
        else:
            name_opt = f'optimizer_{fb}_{n:03}_{i:07}.ckpt'
            state_dict_opt = {"f": self.optimizer["f"].state_dict(), "b": self.optimizer["b"].state_dict()}
        torch.save(state_dict_opt, os.path.join(self.ckpt_dir, name_opt))

    def update_bidir_optimizer_lr(self):
        """Updates the bidirectional optimizer lr at the beginning of the finetuning stage."""
        # update score lr
        self.optimizer.param_groups[0]["lr"] = self.finetune_lr
        if self.use_control_variate:
            # update cv lr
            self.optimizer.param_groups[1]["lr"] = self.finetune_cv_lr

    def get_logger(self, name='logs'):
        return get_logger(self.args, name)

    def get_plotter(self):
        return get_plotter(self, self.args)

    def build_dataloader(self, ds, batch_size, shuffle=True, drop_last=True, repeat=True):
        """Builds a dataloader from dataset ds."""

        def worker_init_fn(worker_id):
            np.random.seed(
                np.random.get_state()[1][0] + worker_id + self.accelerator.process_index * self.args.num_workers)

        dl_kwargs = {"num_workers": self.args.num_workers,
                     "pin_memory": self.args.pin_memory,
                     "worker_init_fn": worker_init_fn}

        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **dl_kwargs)
        dl = self.accelerator.prepare(dl)
        if repeat:
            dl = repeater(dl)
        return dl

    def build_save_dataloaders(self):
        """Builds save dataloaders for test metrics/plots and cache dataloaders for spline fitting."""

        self.test_batch_size = min(int(self.args.test_batch_size) * self.accelerator.num_processes, self.len_test_ds)

        self.save_dls_dict = {}
        self.save_test_init_dl = self.build_dataloader(self.test_init_ds, batch_size=self.test_batch_size,
                                                       shuffle=False, repeat=False, drop_last=False)
        self.save_dls_dict["init"] = self.save_test_init_dl
        self.save_test_final_dl = self.build_dataloader(self.test_final_ds, batch_size=self.test_batch_size,
                                                        shuffle=False, repeat=False, drop_last=False)
        self.save_dls_dict["final"] = self.save_test_final_dl

        # initialization of the cache dataloader for spline fitting
        self.cache_dl = None
        self.cache_init_dl = None
        self.cache_final_dl = None

    def get_sample_net(self, fb):
        """Gets the score network used to simulate the SDE with time direction fb."""
        sample_net = DirectionalScoreWrapper(self.net, fb)
        sample_net = sample_net.to(self.device)
        sample_net.eval()
        return sample_net

    def get_control_point_times(self, sample_direction):
        """
        According to sample_direction, returns (in increasing order):
        - t_space, (N_mean,): grid time of control points
        """
        index_to_save = self.control_point_indexes
        ts = self.sde_ts[sample_direction].clone()
        if sample_direction == 'b':
            ts = torch.flip(ts, dims=[0])
        t_space = ts[index_to_save]
        return t_space

    def train(self):
        """Global function for IMF training.

        Unidirectional setting: it alternates between f and b Markovian projection steps.
        Bidirectional setting: it consists of a pretraining stage and a finetuning stage.
        """
        if self.use_bidir:
            # Bidirectional setting
            if self.checkpoint_stage == "pretraining":
                self.imf_bidir(stage="pretraining")
            self.update_bidir_optimizer_lr()
            self.imf_bidir(stage="finetuning")
        else:
            # Unidirectional setting
            for n in range(self.checkpoint_it, self.n_imf + 1):
                other_dir = 'f' if self.checkpoint_pass == 'b' else 'b'
                if n == self.checkpoint_it:
                    self.imf_iter(self.checkpoint_pass, n)
                    if self.checkpoint_pass == self.starting_direction:
                        self.imf_iter(other_dir, n)
                else:
                    self.imf_iter(self.args.fb_sequence[0], n)
                    self.imf_iter(self.args.fb_sequence[1], n)

    def imf_bidir(self, stage):
        """Performs the Markovian projection of the bidirectional setting at the given stage:
        either pretraining or finetuning."""

        self.accelerator.print(f"Bidirectional training -> stage: {stage}")
        n = 1 if stage == "pretraining" else 2
        start_fb = self.args.bidir.finetune_start_direction

        step = self.step
        first_it = stage == 'pretraining'

        if stage == "pretraining":
            num_epochs = self.pretrain_epochs
            cache_refresh_stride = self.pretrain_refresh_every
        else:
            num_epochs = self.finetune_epochs
            cache_refresh_stride = self.finetune_refresh_every

        num_iter = max(1, (self.len_ds * num_epochs) // self.batch_size)
        refresh_every = max(1, (self.len_ds * cache_refresh_stride) // self.batch_size)

        max_iter_annealing = (self.len_ds * min(int(self.max_epoch_annealing), num_epochs)) // self.batch_size

        self.set_seed(
            seed=self.compute_current_step(step, n,
                                           num_iter) * self.accelerator.num_processes + self.accelerator.process_index)

        bar_iter = tqdm(range(step, num_iter + 1))
        for i in bar_iter:
            # the annealing strategy is only active at the pretraining stage
            factor_state_cost = (
                min(i / max_iter_annealing, self.factor_state_cost)
                if self.use_state_cost_annealing and stage == "pretraining"
                else self.factor_state_cost)
            refresh_index = (i - 1) // refresh_every

            # we alternate the time direction depending on refresh_every
            fb = start_fb if refresh_index % 2 == 0 else ("b" if start_fb == "f" else "f")

            if i == step or (i - 1) % refresh_every == 0:
                # build the cache dataloader
                torch.cuda.empty_cache()
                self.build_cache_dataloader(fb, n, first_it=first_it)

            self.net.train()
            if self.use_control_variate:
                self.cv_net["f"].train()
                self.cv_net["b"].train()

            self.set_seed(
                seed=self.compute_current_step(i, n,
                                               num_iter) * self.accelerator.num_processes + self.accelerator.process_index)

            if stage == "pretraining":
                # here, we run the forward and backward Markovian projections together
                imf_loss, metrics = self.get_bidir_train_loss(factor_state_cost=factor_state_cost, stage=stage)
            else:
                # here, we run solely the fb Markovian projection
                imf_loss, metrics = self.get_unidir_train_loss(fb, factor_state_cost=factor_state_cost, stage=stage)

            self.accelerator.backward(imf_loss)

            if i == 1 or i % self.stride_log == 0 or i == num_iter:
                # train logs : we restart 'step' from 0 at finetuning when logging
                self.logger.log_metrics(metrics, step=self.compute_current_step(i, 1, num_iter))

            bar_iter.set_description(f"Stage : {stage}, IMF loss={imf_loss:.2f}")

            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

            if i != num_iter:
                self.save_step(i, num_iter, n, stage=stage,
                               force_test=(stage == "finetuning" and i % refresh_every == 0), )

        self.save_step(num_iter, num_iter, n, stage=stage)

        self.clear()
        self.step = 1  # to start again a novel IMF step

    def imf_iter(self, fb, n):
        """Performs the fb Markovian projection at outer iteration n."""

        self.accelerator.print(
            'Unidirectional training -> IMF iteration: ' + str(n) + '/' + str(self.n_imf) + ', dir:' + fb)

        step = self.step
        first_it = (n == 1) and (fb == self.starting_direction)

        num_iter = (self.len_ds * self.num_epochs) // self.batch_size
        cache_refresh_stride = (
            int(self.first_it_refresh_every)
            if n == 1
            else int(self.next_it_refresh_every))
        refresh_every = max(1, (self.len_ds * cache_refresh_stride) // self.batch_size)

        max_iter_annealing = (self.len_ds * min(int(self.max_epoch_annealing), self.num_epochs)) // self.batch_size

        self.set_seed(seed=self.compute_current_step(step,
                                                     n,
                                                     num_iter) * self.accelerator.num_processes + self.accelerator.process_index)

        bar_iter = tqdm(range(step, num_iter + 1))
        for i in bar_iter:
            # the annealing strategy is only active at the first outer iteration
            factor_state_cost = min(i / max_iter_annealing, self.factor_state_cost) \
                if self.use_state_cost_annealing and n == 1 \
                else self.factor_state_cost

            if (i == step) or (i - 1) % refresh_every == 0:
                # build the cache dataloader
                torch.cuda.empty_cache()
                self.build_cache_dataloader(fb, n, first_it=first_it)

            self.net.train()
            if self.use_control_variate:
                self.cv_net[fb].train()

            self.set_seed(
                seed=self.compute_current_step(i, n,
                                               num_iter) * self.accelerator.num_processes + self.accelerator.process_index)

            imf_loss, metrics = self.get_unidir_train_loss(fb, factor_state_cost=factor_state_cost, n=n)

            self.accelerator.backward(imf_loss)

            if i == 1 or i % self.stride_log == 0 or i == num_iter:
                # train logs
                self.logger.log_metrics(metrics, step=self.compute_current_step(i, n, num_iter))

            bar_iter.set_description(f"IMF loss={imf_loss:.2f}")

            self.optimizer[fb].step()
            self.optimizer[fb].zero_grad(set_to_none=True)

            if i != num_iter:
                self.save_step(i, num_iter, n, fb=fb)

        self.save_step(num_iter, num_iter, n, fb=fb)

        self.clear()
        self.step = 1  # to start again a novel IMF step

    def save_step(self, i, num_iter, n, fb='f', stage='pretraining', force_test=False, ):
        """Activates checkpoint saving or test procedure."""
        if i == num_iter or i % self.stride_ckpt == 0:
            # save checkpoints
            self.save_ckpt(i, n, fb=fb, stage=stage)
        if force_test or i == num_iter or i % self.stride == 0:
            # run test metrics and do the plots
            self.plot_and_test_step(i, n, num_iter, fb=fb, stage=stage)

    def plot_and_test_step(self, i, n, num_iter, fb='f', stage='pretraining'):
        """Run test metrics and do the plotting."""
        self.set_seed(seed=0 + self.accelerator.process_index)
        self.accelerator.print('Running test metrics and plotting...')
        if self.use_bidir:
            for fb_ in ["f", "b"]:
                test_metrics = self.plotter(i, n, fb_, name_i=f"{stage}_{i}")
                test_metrics["stage"] = stage

                if self.accelerator.is_main_process:
                    # test logs : we restart 'step' from 0 at finetuning when logging
                    self.save_logger.log_metrics(test_metrics, step=self.compute_current_step(i, 1, num_iter))
        else:
            test_metrics = self.plotter(i, n, fb)
            if self.accelerator.is_main_process:
                # test logs
                self.save_logger.log_metrics(test_metrics, step=self.compute_current_step(i, n, num_iter))
        self.accelerator.print('Running test metrics and plotting: DONE !')

    def get_unidir_train_loss(self, fb, factor_state_cost=1.0, n=None, stage=None):
        """[MARKOVIAN PROJECTION]
        Computes the fb Markovian projection loss.

        This is used in both unidirectional and bidirectional settings."""
        batch = next(self.cache_dl)

        if self.args.cost_name in DISC_STATE_COSTS:
            t, s, cv_s_times, coeff_cv_s = self.get_train_discrete_times(fb)
        else:
            t, s = self.get_train_continuous_times(fb)
            cv_s_times = None
            coeff_cv_s = None

        sample_direction = "b" if fb == "f" else "f"

        imf_loss, imf_loss_var, target_var = self.get_train_loss(batch, t, fb, sample_direction, s=s,
                                                                 cv_s_times=cv_s_times,
                                                                 coeff_cv_s=coeff_cv_s,
                                                                 factor_state_cost=factor_state_cost, )
        metrics = {
            "fb": fb,
            "imf_loss": imf_loss,
            "imf_loss_var": imf_loss_var,
            "target_var": target_var,
        }
        if n is not None:
            metrics["imf_iter"] = n
        if stage is not None:
            metrics["stage"] = stage

        return imf_loss, metrics

    def get_bidir_train_loss(self, factor_state_cost=1., stage='pretraining'):
        """[MARKOVIAN PROJECTION]
        Computes the mean of the forward and backward Markovian projection losses.

        This is only used in the bidirectional setting."""

        batch = next(self.cache_dl)
        B = batch["x0"].shape[0]
        assert B % 2 == 0, 'Batch size should be divisible by 2'
        b = B // 2

        batch_f = {k: v[:b] for k, v in batch.items()}
        batch_b = {k: v[b:] for k, v in batch.items()}

        if self.args.cost_name in DISC_STATE_COSTS:
            t_f, s_f, cv_s_times_f, coeff_cv_s_f = self.get_train_discrete_times("f", batch_size=batch_f["x0"].shape[0])
            t_b, s_b, cv_s_times_b, coeff_cv_s_b = self.get_train_discrete_times("b", batch_size=batch_b["x0"].shape[0])
        else:
            t_f, s_f = self.get_train_continuous_times("f", batch_size=batch_f["x0"].shape[0])
            t_b, s_b = self.get_train_continuous_times("b", batch_size=batch_b["x0"].shape[0])
            cv_s_times_f = cv_s_times_b = None
            coeff_cv_s_f = coeff_cv_s_b = None

        imf_loss_f, imf_loss_var_f, target_var_f = self.get_train_loss(batch_f, t_f, "f", "b", s=s_f,
                                                                       cv_s_times=cv_s_times_f,
                                                                       coeff_cv_s=coeff_cv_s_f,
                                                                       factor_state_cost=factor_state_cost, )

        imf_loss_b, imf_loss_var_b, target_var_b = self.get_train_loss(batch_b, t_b, "b", "f", s=s_b,
                                                                       cv_s_times=cv_s_times_b,
                                                                       coeff_cv_s=coeff_cv_s_b,
                                                                       factor_state_cost=factor_state_cost, )

        imf_loss = 0.5 * (imf_loss_f + imf_loss_b)

        metrics = {
            "imf_loss": imf_loss,
            "stage": stage,
            "imf_loss_f": imf_loss_f.detach(),
            "imf_loss_b": imf_loss_b.detach(),
            "imf_loss_var_f": imf_loss_var_f,
            "imf_loss_var_b": imf_loss_var_b,
            "target_var_f": target_var_f,
            "target_var_b": target_var_b,
        }

        return imf_loss, metrics

    def build_cache_dataloader(self,
                               fb,
                               n,
                               first_it=False):
        """ [RECIPROCAL PROJECTION]
        Builds a dataloader by running the spline fitting step in direction forward_or_backward at iteration n.

        If first_it : initializes the control points with the linear interpolant
        Else: initializes the control points by sampling from the previous learned SDE at the mean control times

        Then, applies the variational optimization (this stage is skipped in DSBM).
        """
        sample_direction = "b" if fb == "f" else "f"
        sample_net = self.get_sample_net(sample_direction)
        control_point_times = self.get_control_point_times(
            sample_direction)  # control times are always given in increasing order

        # Initial dataset -> build new dataloader
        self.cache_init_dl = iter(
            self.build_dataloader(self.init_ds, batch_size=self.spline_batch_size, repeat=False, drop_last=True))

        # Final dataset -> build new dataloader
        self.cache_final_dl = iter(
            self.build_dataloader(self.final_ds, batch_size=self.spline_batch_size, repeat=False, drop_last=True))

        num_batches = self.len_ds // self.spline_batch_size

        self.accelerator.print(f"Building Conditional Sampler from {num_batches} data batches...")
        cond_outputs = []
        start = time.time()

        for index_batch in range(num_batches):
            x0 = next(self.cache_init_dl)[0]
            x1 = next(self.cache_final_dl)[0]
            B = x0.shape[0]
            if first_it:
                # Unidirectional : very first training direction // Bidirectional : pretraining stage
                # Linear interpolant to initialize the control points
                control_point_times_ = control_point_times.view(1, self.N_mean, 1)
                coeff0 = 1. - control_point_times_ / self.T
                x_t = coeff0 * x0.view(B, 1, *self.shape_x) + (1. - coeff0) * x1.view(
                    B, 1, *self.shape_x)
            else:
                # Compute the samples from previous SDE to initialize the control points
                x_start = x0 if sample_direction == 'f' else x1
                with torch.no_grad():
                    x_t, _, _ = self.langevin.sample_sde(x_start, sample_net, sample_direction,
                                                         self.sde_ts, self.control_point_indexes)
                # Update the terminal state with the output of SDE simulation
                if sample_direction == 'f':
                    x1 = x_t[:, -1]
                else:
                    x0 = x_t[:, -1]

                # Put the trajectory forward in time if needed
                if sample_direction == "b":
                    x_t = torch.flip(x_t, dims=[1])

            assert x_t.shape == (B, self.N_mean, *self.shape_x)

            # Create and fit the conditional sampler with spline model (Reciprocal projection)
            # -> nothing happens here in DSBM
            cond_output = self.fit_conditional_sampler(x0, x1, x_t, control_point_times, fb, n,
                                                       first_it, index_batch)

            cond_outputs.append(cond_output)
            # save the learned intermediate std control points to initialize the future splines
            if "gamma_xs" in cond_output:
                if not hasattr(self, "prev_gamma_xs"):
                    self.prev_gamma_xs = {}
                self.prev_gamma_xs[fb] = cond_output["gamma_xs"].detach().cpu().clone()

        cache_ds = self.get_conditional_dataset(cond_outputs)
        cache_dl = self.build_dataloader(cache_ds, self.batch_size)
        self.cache_dl = cache_dl

        self.clear()

        stop = time.time()
        self.accelerator.print(f"Conditional Sampler built from {num_batches} data batches!")
        self.accelerator.print(f"Cache load time: {stop - start: .2f} s")

    def get_train_continuous_times(self, fb, t_input=None, batch_size=None):
        """ Sampling time method designed for continuous-time state cost functions.

        Returns sampled times to compute the Markovian projection loss in direction fb:
        - t: (batch_size, 1) : times associated to the velocity field
        - s: (batch_size, num_s_per_t, 1) of dtype float : times associated to the state cost
            - if fb=='f': t < s < T
            - if fb=='b': 0 < s < t
            - if not self.sample_s: None (DSBM/GSBM case)

        If t_input is not None, t=t_input.
        """
        # 1. sampling t (low-discrepancy sampler)
        if t_input is None:
            batch_size = self.batch_size if batch_size is None else batch_size
            u = torch.rand((1, 1), device=self.device)
            i = torch.arange(batch_size, device=self.device).view(batch_size, 1) / batch_size
            t = torch.remainder(u + i, 1.0)
            t = 2 * self.eps + (self.T - 4 * self.eps) * t
            # uniform sampling on (2*eps,T-2*eps)
        else:
            # t times are already provided
            t = t_input
        batch_size = t.shape[0]
        t_ = t.view(batch_size, 1, 1)

        # 2. sampling s|t (low-discrepancy sampler)
        if fb == 'f' and self.sample_s:
            u = torch.rand((1, self.num_s_per_t, 1), device=t.device)
            i = torch.arange(batch_size, device=t.device).view(batch_size, 1, 1) / batch_size
            s_unit = torch.remainder(u + i, 1.0)
            s = t_ + self.eps + (self.T - t_ - 2 * self.eps) * s_unit
            # uniform sampling on (t+eps,T-eps)
        elif fb == 'b' and self.sample_s:
            u = torch.rand((1, self.num_s_per_t, 1), device=t.device)
            i = torch.arange(batch_size, device=t.device).view(batch_size, 1, 1) / batch_size
            s_unit = torch.remainder(u + i, 1.0)
            s = self.eps + (t_ - 2 * self.eps) * s_unit
            # uniform sampling on (eps,t-eps)
        else:
            # DSBM/GSBM case
            s = None
        return t, s

    def get_train_discrete_times(self, fb, t_input=None, batch_size=None):
        """Sampling time method designed for discrete-time state cost functions.

        Returns sampled times to compute the Markovian projection loss in direction fb:
        - t: (batch_size, 1) : times associated to the velocity field
        - s: (batch_size, num_s_per_t, 1) of dtype long : indexes associated to the state cost
            -> in 'f' : cost time indexes k such that s_k > t, values in {0, ...K-1, K} (K means no valid cost)
            -> in 'b' : cost time indexes k such that s_k < t, values in {-1, 0, ...K-1} (-1 means no valid cost)
        - cv_s_times: (batch_size, num_s_per_t, 1) of dtype float : times associated to CV
            -> in 'f' : uniformly sampled between s-1 and s (min clamp = t), or t and T if s=K
            -> in 'b' : uniformly sampled between s and s+1 (max clamp = t), or 0 and t if s=-1
        - coeff_cv_s: (batch_size, num_s_per_t, 1): loss coefficients associated to cv_s_times
            -> valid even when s=-1 or s=K

        If t_input is not None, t=t_input.
        If not self.sample_s, s is None.
        If not self.use_control_variate, cv_s_times and coeff_cv_s are None.
        """
        assert self.disc_cost_times is not None
        # 1. sampling t (low-discrepancy sampler)
        if t_input is None:
            batch_size = self.batch_size if batch_size is None else batch_size
            u = torch.rand((1, 1), device=self.device)
            i = torch.arange(batch_size, device=self.device).view(batch_size, 1) / batch_size
            t = torch.remainder(u + i, 1.0)
            t = 2 * self.eps + (self.T - 4 * self.eps) * t
            # uniform sampling on (2*eps,T-2*eps)
        else:
            # t times are already provided
            t = t_input
        batch_size = t.shape[0]
        t_ = t.view(batch_size, 1, 1).expand(-1, self.num_s_per_t, -1)

        # 2. sampling s_k|t :
        # if 'f' : uniformly sample s_k > t excluding time=T (index = K)
        # if 'b' : uniformly sample s_k < t excluding time=0 (index = -1)

        i = torch.arange(batch_size, device=t.device).view(batch_size, 1, 1) / batch_size
        u = torch.rand((1, self.num_s_per_t, 1), device=t.device)
        u = torch.remainder(u + i, 1.0)

        if fb == 'f' and self.sample_s:
            min_indexes = torch.searchsorted(self.disc_cost_times, t + 2 * self.eps, right=True)
            # (B,1), equal to K if boundary
            range_indexes = self.K - min_indexes  # in [0 , ..., K]
            s_indexes = min_indexes[:, None, :] + torch.floor(u * range_indexes[:, None, :]).long()
            # (B,S,1), in [0, 1, ..., K]
            assert s_indexes.shape == (batch_size, self.num_s_per_t, 1)
        elif fb == 'b' and self.sample_s:
            max_indexes = torch.searchsorted(self.disc_cost_times, t - 2 * self.eps, right=False)
            # (B,1), equal to 0 if boundary
            valid = (max_indexes > 0).view(batch_size, 1, 1).expand(-1, self.num_s_per_t, -1)
            s_indexes = torch.full((batch_size, self.num_s_per_t, 1), -1, dtype=torch.long, device=t.device)
            s_indexes[valid] = torch.floor(u * max_indexes[:, None, :]).long()[valid]
            # (B,S,1), in [-1, 0, ..., K-1]
            assert s_indexes.shape == (batch_size, self.num_s_per_t, 1)
        else:
            s_indexes = None

        # 3. sampling s|s_k if using CV :(low-discrepancy sampler)
        # if 'f' : uniform sampling on (s_k-1(t),s_k(t))
        # if 'b' : uniform sampling on (s_k(t),s_k+1(t))
        if self.use_control_variate:
            if fb == 'f':
                # Lower bound
                lower = torch.maximum(self.disc_cost_times[(s_indexes - 1).clamp(min=0)], t_)
                lower[s_indexes == 0] = t_[s_indexes == 0]
                # Upper bound: if s_indexes == K (no associated cost) -> T, else -> cost_times[s_indexes]
                upper = torch.where(s_indexes == self.K,
                                    torch.full_like(s_indexes, self.T, dtype=t.dtype),
                                    self.disc_cost_times[s_indexes.clamp(max=self.K - 1)])
                # works for any t, even t with no associated time cost
            else:
                # Lower bound: if s_indexes == -1 (no associated cost) -> 0, else -> cost_times[s_indexes]
                lower = torch.where(s_indexes == -1,
                                    torch.full_like(s_indexes, 0., dtype=t.dtype),
                                    self.disc_cost_times[s_indexes.clamp(min=0)])

                # Upper bound
                upper = torch.minimum(self.disc_cost_times[(s_indexes + 1).clamp(max=self.K - 1)], t_)
                upper[s_indexes == self.K - 1] = t_[s_indexes == self.K - 1]
                # works for any t, even t with no associated time cost
            assert lower.shape == (batch_size, self.num_s_per_t, 1)
            assert upper.shape == (batch_size, self.num_s_per_t, 1)
            r_s = torch.rand((1, self.num_s_per_t, 1), device=t.device)
            r_s = torch.remainder(r_s + i, 1.0)
            cv_s_times = lower + self.eps + (upper - lower - 2 * self.eps) * r_s
            # coeff for loss : valid for t with no associated cost
            coeff_cv_s = self.get_num_cost_times_per_t(t[:, None, :], fb).clamp(min=1) * (upper - lower)
        else:
            cv_s_times = None
            coeff_cv_s = None

        return t, s_indexes, cv_s_times, coeff_cv_s

    def get_train_loss(self, batch, t, forward_or_backward, sample_direction,
                       s=None,
                       cv_s_times=None,
                       coeff_cv_s=None,
                       factor_state_cost=1.0, ):
        raise NotImplementedError

    def fit_conditional_sampler(self, x0, x1, x_t, control_point_times, forward_or_backward, n, first_it, index_batch):
        raise NotImplementedError

    def get_conditional_dataset(self, cond_outputs):
        raise NotImplementedError


class DSBM(TSBM):
    """DSBM instance of the generalized SB problem."""

    def __init__(self, init_ds, final_ds, test_init_ds, test_final_ds, shape_x, args, accelerator=None):
        super().__init__(init_ds, final_ds, test_init_ds, test_final_ds, shape_x, args, accelerator)

        self.sample_s = False
        self.use_state_cost_annealing = False
        self.factor_state_cost = 1.

        # 2 control points: t=0 and t=T
        self.N_mean = 2

        # Build control times
        self.build_control_times()

        # Build Langevin sampler
        self.langevin = Twisted_BM_ZeroCost(self.T, self.sigma, self.shape_x, self.loss_param, self.device)

    def fit_conditional_sampler(self, x0, x1, x_t, control_point_times, fb, n, first_it, index_batch):
        """
        Performs the training of the Reciprocal projection (nothing in this case).

        - x0: (B, dim)
        - x1: (B, dim)
        - x_t: (B, N_mean, dim)
        - control_point_times: (N_mean, )
        - fb: 'f' or 'b'
        - n: int
        - first_it: True or False
        - index_batch: int

        Returns:
        - output:
            - x0: (B, dim)
            - x1: (B, dim)
         """

        output = {'x0': x0.detach().cpu().clone(), 'x1': x1.detach().cpu().clone()}

        return output

    def get_conditional_dataset(self, cond_outputs):
        x0 = gather(cond_outputs, 'x0')
        x1 = gather(cond_outputs, 'x1')
        return ZeroCostDataset(x0=x0, x1=x1)

    def get_train_samples_from_bridge(self, x0, x1, t):
        """
        Computes training samples to be used in the Markovian projection loss.

        - x0: (B, dim)
        - x1: (B, dim)
        - t: (B, 1)

        Returns:
            - z_t: (B,dim) -> computed as z_t|x_0,x_1 (tractable)
            - z_noise_t: (B,dim) -> standard Gaussian noise associated to z_t
         """
        # computing z_t and z_noise_t
        z_t = (t / self.T) * x1 + (1. - t / self.T) * x0
        z_noise_t = torch.randn_like(z_t)
        z_t += self.sigma * torch.sqrt(t * (1. - t / self.T)) * z_noise_t
        return z_t, z_noise_t

    def get_train_loss(self, batch, t, fb, sample_direction,
                       s=None,
                       cv_s_times=None,
                       coeff_cv_s=None,
                       factor_state_cost=1.0, ):
        """
        Computes an empirical estimator of the Markovian projection loss.

        - batch: dict -> 'x0', 'x1'
        - t: (B, 1)
        - fb: 'f' or 'b'
        - sample_direction: if fb=='f' -> 'b', if fb=='b' -> 'f'
        - s: None (not used in this function)
        - cv_s_times: None (not used in this function)
        - coeff_cv_s: None (not used in this function)
        - factor_state_cost: 1.0 (not used in this function)

        Returns:
            - imf_loss: scalar -> IMF loss averaged on the batch
            - imf_loss_var : scalar -> variance of IMF loss with respect to s (0. here, no s)
            - target_var : scalar -> variance of IMF target with respect to s (0. here, no s)
         """

        x0, x1 = batch['x0'], batch['x1']

        z_t, z_noise_t = self.get_train_samples_from_bridge(x0, x1, t)

        target = self.langevin.get_train_target_from_bridge(x0, x1, z_noise_t, t, fb)
        pred = self.net(z_t, t, fb)

        assert target.shape == pred.shape

        # Define loss rescaling as in the original DSBM paper
        if self.loss_param == 'velocity' and fb == 'f':
            loss_scale = torch.sqrt(1. + self.sigma ** 2 * (t / self.T) / (self.T - t))
        elif self.loss_param == 'velocity' and fb == 'b':
            loss_scale = torch.sqrt(1. + self.sigma ** 2 * (1. - t / self.T) / t)
        else:
            loss_scale = 1.

        # Compute IMF loss
        sq_errors = ((pred - target) / loss_scale) ** 2
        imf_loss = sq_errors.mean()

        return imf_loss, 0., 0.


class TSBM_Spline(TSBM):
    """GSBM/TSBM instance of the generalized SB problem with spline bridge parameterization."""

    def __init__(self, init_ds, final_ds, test_init_ds, test_final_ds, shape_x, args, accelerator=None):
        super().__init__(init_ds, final_ds, test_init_ds, test_final_ds, shape_x, args, accelerator)

        # Control points for spline approximation
        self.N_mean = self.args.spline.N_mean
        assert self.N_mean > 2
        self.N_std = self.args.spline.N_std
        assert self.N_std > 2

        # Type of GSB method
        self.gsb_method = self.args.gsb_method
        assert self.gsb_method in ['tsbm', 'gsbm']
        if self.gsb_method == 'gsbm':
            self.sample_s = False
            self.use_control_variate = False
            self.grid_ys = None
            self.use_state_cost_annealing = False
            self.factor_state_cost = 1.
        else:
            assert self.num_s_per_t > 0, 'Input num_s_per_t must be greater than 1'
            self.grid_ys = torch.linspace(0.0, self.T, self.args.spline.n_grid, device=self.device)
            assert 0. <= self.factor_state_cost, 'The annealing maximum ratio should be set greater than 0.'

        # Build control times
        self.build_spline_control_times()

        # Build Langevin sampler
        self.langevin = Twisted_BM_GeneralCost(self.V, self.gsb_method, self.T, self.sigma, self.shape_x,
                                               self.loss_param,
                                               self.device)

    def build_spline_control_times(self):
        """Builds mean and std control times."""
        # Build control times for spline means
        self.build_control_times()
        # Build control times for spline stds
        self.control_std_times = torch.linspace(0, self.T, self.N_std, device=self.device)

    @torch.no_grad()
    def get_train_samples_from_spline(self, t, fb, gpath, s_times=None, mask_s=None, cv_s_times=None):
        """
        Computes training samples from the learned spline to be used in the Markovian projection loss.

        - t: (B,)
        - fb: 'f' or 'b'
        - gpath: Gaussian VI module associated to the current batch
        - s_times: (B,S) : cost times associated to t or None (GSBM)
        - mask_s: (B,S) : whether s_times is valid (useful for discrete-time TSBM) or None (GSBM)
        - cv_s_times: (B,S) or None (continuous-time TSBM & GSBM)

        --> z_t: (B, dim) -> computed as z_t|x_0,x_1
        --> z_s: (B, S, dim) : None if s_times is None
            - if fb=='f' and cv_s_times is None : computed as z_s|z_t,x_1
            - if fb=='b' and cv_s_times is None : computed as z_s|x_0,z_t
            - if fb=='f' and cv_s_times is not None : computed as z_s|z_cv_s,x_1 (by construction, cv_s_times<s_times)
            - if fb=='b' and cv_s_times is not None : computed as z_s|x_0,z_cv_s (by construction, cv_s_times>s_times)
            -> on ~mask_s : z_s=z_t
        --> z_cv_s: (B, S, dim) : None if cv_s_times is None
            - if fb=='f' and cv_s_times is None : computed as z_cv_s|z_t,x_1
            - if fb=='b' and cv_s_times is None : computed as z_cv_s|x_0,z_t
        """
        B = t.shape[0]

        # 1. Sampling z_t : common to all methods
        z_t = gpath.sample_xt(t, 1).detach()[torch.arange(B), 0, torch.arange(B)]  # (B,dim)
        assert z_t.shape == (B, *self.shape_x)

        z_s = None
        z_cv_s = None

        # 2. Sampling z_s|z_t : only used in TSBM
        if s_times is not None:
            assert s_times.shape[0] == B
            S = s_times.shape[1]

            t_ = t.unsqueeze(-1).expand(-1, S)
            z_t_ = z_t.unsqueeze(1).expand(-1, S, -1)

            if cv_s_times is not None:
                # 2.a sampling z_cv_s| z_t
                assert cv_s_times.shape == (B, S)
                z_cv_s = gpath.sample_s_given_t(t.unsqueeze(1),
                                                z_t.unsqueeze(1),
                                                cv_s_times.unsqueeze(1),
                                                fb).detach().squeeze(1)  # (B,S,dim)
            else:
                # z_cv_s=z_t (continuous case & no-CV discrete case)
                z_cv_s = z_t_  # (B,S,dim)
                cv_s_times = t_.clone()  # (B,S)

            assert z_cv_s.shape == (B, S, *self.shape_x)

            # 2.b sampling z_s| z_cv_s
            z_s = gpath.sample_s_given_t(cv_s_times,
                                         z_cv_s,
                                         s_times.unsqueeze(-1),
                                         fb).detach().squeeze(2)  # (B,S,dim)

            z_s[~mask_s] = z_t_[~mask_s]  # z_s=z_t out of the mask
            assert z_s.shape == (B, S, *self.shape_x)

            if cv_s_times is None:
                z_cv_s = None

        return z_t, z_s, z_cv_s

    def fit_conditional_sampler(self, x0, x1, x_t, control_point_times, fb, n, first_it, index_batch):
        """
        Computes an empirical estimator of the Markovian projection loss.

        - x0: (B, dim)
        - x1: (B, dim)
        - x_t: (B, N_mean, dim)
        - control_point_times: (N_mean, )
        - fb: 'f' or 'b'
        - n: int
        - first_it: True or False
        - index_batch: int

        Returns:
        - output:
            - mean_xt: (B, N_mean, dim) -> means of control points
            - gamma_xs: (B, N_std, 1) -> stds of control points

         """
        B = x_t.shape[0]
        # reuse the saved std values of the spline for initialization, otherwise initialize to 0
        prev_gamma_xs = getattr(self, "prev_gamma_xs", {}).get(fb)
        if prev_gamma_xs is None:
            std_s = torch.zeros(B, self.N_std, 1, device=self.device)
        else:
            prev_gamma_xs = prev_gamma_xs.to(self.device)
            if prev_gamma_xs.shape[0] >= B:
                std_s = prev_gamma_xs[:B].clone()
            else:
                repeats = math.ceil(B / prev_gamma_xs.shape[0])
                std_s = prev_gamma_xs.repeat(repeats, 1, 1)[:B].clone()

        basedrift = BrownianBridgeDrift() if self.gsb_method == 'tsbm' else ZeroBaseDrift()
        # Create the variational approximation path
        gpath = gpath_lib.EndPointGaussianPath(t=control_point_times, xt=x_t, s=self.control_std_times, ys=std_s,
                                               sigma=self.sigma, t_final=self.T, basedrift=basedrift,
                                               device=self.device, grid_ys=self.grid_ys)
        gpath.train()

        # Optimize the spline
        loss_fn = gpath_lib.build_loss_fn(gpath=gpath, V=self.V, discrete_setting=self.discrete_cfg)

        with torch.enable_grad():
            # In the pretraining stage of the bidirectional setting, we optimize along both directions
            vi_fb = "fb" if first_it and self.use_bidir else fb

            vi_results = gpath_lib.fit(ccfg=self.args.spline, gpath=gpath, fb=vi_fb,
                                       loss_fn=loss_fn, cost_name=self.args.cost_name, discrete_cfg=self.discrete_cfg,
                                       first_it=first_it, eps=self.eps, verbose=True, )

        # Save the plots in 2D
        if self.args.dim == 2:
            self.plotter.plot_gpath_2d(vi_results, n, fb, index_batch)

        # Build output
        xt = gpath.mean.xt.detach().cpu().clone()
        ys = gpath.gamma.xt.detach().cpu().clone()
        output = {'mean_xt': xt, 'gamma_xs': ys}

        if self.gsb_method == 'tsbm':
            # save the 1D grid of 1/gamma^2 for the batch : accelerate bridge sampling !
            gpath.gamma.build_grid()
            F_grid = gpath.gamma.F_grid.detach().cpu().clone()
            output['F_grid_xs'] = F_grid

        return output

    def get_conditional_dataset(self, cond_outputs):
        """Builds the dataset from the mean and std control points."""
        mean_xt = gather(cond_outputs, 'mean_xt')
        gamma_xs = gather(cond_outputs, 'gamma_xs')
        if self.gsb_method == 'tsbm':
            F_grid_xs = gather(cond_outputs, 'F_grid_xs')
        else:
            F_grid_xs = None

        return SplineDataset(mean_xt=mean_xt, gamma_xs=gamma_xs, F_grid_xs=F_grid_xs)

    def get_train_loss(self, batch, t, fb, sample_direction,
                       s=None,
                       cv_s_times=None,
                       coeff_cv_s=None,
                       factor_state_cost=1.0, ):
        """
        Computes an empirical estimator of the Markovian projection loss.

        - batch: dict -> 'x0', 'x1', 'mean_xt', 'gamma_xs', 'F_grid_xs'
        - t: (B, 1)
        - fb: 'f' or 'b'
        - sample_direction: if fb=='f' -> 'b', if fb=='b' -> 'f'
        - first_it: True or False
        - s: (B, S, 1) -> times if continuous-time cost, indexes if discrete-time cost
        - cv_s_times: (B, S, 1) if discrete-time cost V & CV, None else
        - coeff_cv_s: (B, S, 1) if discrete-time cost V & CV, None else
        - factor_state_cost: float
        - return_sq_errors: True or False

        Returns:
            - imf_loss: scalar -> IMF loss averaged on the batch
            - imf_loss_var : scalar -> variance of IMF loss with respect to s (0. for GSBM)
            - target_var : scalar -> variance of IMF target with respect to s (0. for GSBM)
         """

        x0, x1 = batch['x0'], batch['x1']
        F_grid_ys = batch.get('F_grid_xs')

        B = x0.shape[0]

        # Create the variational approximation path on the batch
        control_point_times = self.get_control_point_times(sample_direction)
        basedrift = BrownianBridgeDrift() if self.gsb_method == 'tsbm' else ZeroBaseDrift()
        gpath = gpath_lib.EndPointGaussianPath(t=control_point_times, xt=batch['mean_xt'], s=self.control_std_times,
                                               ys=batch['gamma_xs'], sigma=self.sigma, t_final=self.T,
                                               basedrift=basedrift, device=self.device,
                                               grid_ys=self.grid_ys, F_grid_ys=F_grid_ys)

        gpath.eval()

        assert t.shape == (B, 1)
        if s is not None:
            assert s.shape[0] == B
            S = s.shape[1]
        else:
            S = 1

        if s is not None and self.args.cost_name in DISC_STATE_COSTS:
            # discrete time cost function : s are indexes that may contain -1 ('b') or K ('f')
            mask_s = (s >= 0) & (s < self.K)  # valid cost times to evaluate
            s = s.clamp(min=0, max=self.K - 1).long()
            s_times = self.disc_cost_times[s]  # we evaluate the cost on the first or last cost in boundary cases
        elif s is not None:
            # continuous time cost function
            s_times = s.clone()
            mask_s = torch.full_like(s, True, dtype=torch.bool)
        else:
            # GSBM
            s_times = None
            mask_s = None

        # Sample from gpath : zt|x0,x1 & zs|zt,x1 if 'f' or zs|x0,zt if 'b' & z_cv_s if needed
        z_t, z_s, z_cv_s = self.get_train_samples_from_spline(t.squeeze(-1), fb, gpath,
                                                              s_times.squeeze(-1) if s_times is not None else None,
                                                              mask_s.squeeze(-1) if mask_s is not None else None,
                                                              cv_s_times.squeeze(
                                                                  -1) if cv_s_times is not None else None)
        # z_t: (B,dim)
        # z_s: (B,S,dim) if not None
        # z_cv_s: (B,S,dim) if not None

        # Compute the target and the prediction
        target = self.langevin.get_train_target_from_cond_sampler(x0, x1, z_t, t, fb, gpath, z_s, s, s_times,
                                                                  mask_s, z_cv_s, cv_s_times, coeff_cv_s,
                                                                  self.cv_net[fb], factor_state_cost, )

        t_ = t.expand(-1, S).reshape(B * S)
        mask_cost = (t_ / self.T < self.V.max_t_cost) if fb == 'f' else (t_ / self.T > self.V.min_t_cost)

        if self.args.cost_name in DISC_STATE_COSTS and self.use_control_variate and self.gsb_method == 'tsbm':
            # discrete time + CV : corrective term due to the mismatch between cost times and boundary times 0/T
            target_boundary = self.compute_corrective_cv_term(x0, x1, t, z_s, s_times, mask_s, gpath, fb)
            target_boundary[~mask_cost] = torch.zeros_like(target)[~mask_cost]
        else:
            target_boundary = torch.zeros_like(target)
        target += target_boundary

        pred = self.net(z_t, t, fb)

        if self.gsb_method == 'tsbm':
            minus_ones = (-1,) * len(self.shape_x)
            pred = pred.unsqueeze(1).expand(-1, S, *minus_ones).reshape(B * S, *self.shape_x)
            effective_S = S
        else:
            effective_S = 1

        assert target.shape == pred.shape

        # Compute IMF loss
        sq_errors = (pred - target) ** 2
        imf_loss = sq_errors.mean()

        # Compute the target variance term along s: 0 in the case of GSBM
        target = target.reshape(B, effective_S, *self.shape_x)
        target_var = target ** 2 - (target.mean(dim=1, keepdim=True)) ** 2

        # Compute the variance of the loss along s: 0 in the case of GSBM
        imf_loss_var = sq_errors.mean(dim=tuple(range(-len(self.shape_x), 0))).var(unbiased=False).detach()

        return imf_loss, imf_loss_var, target_var.mean().detach()

    def compute_corrective_cv_term(self, x0, x1, t, z_s, s_times, mask_s, gpath, fb):
        """
        Computes the corrective control variate TSBM target term for discrete-time cost.

        - t: (B,1)
        - z_s: (B,S,D) : particles associated to cost times -> we have z_s[~mask_s]=z_t[~mask_s]
        - s_times: (B,S,1) : cost times associated to z_s, with values in {0, ..., self.K}
        - mask_s: (B,S,1) : whether z_s is valid (excludes invalid s_times)
        - gpath: gaussian VI module associated to the current batch
        - fb: 'f' or 'b'

        --> target_boundary: (B*S,D) -> corrective term for CV in discrete time (0 out of mask_s)
        """
        assert self.disc_cost_times is not None, 'This corrective term should only be computed for CV+discrete-time cost.'
        last_cost_time = self.disc_cost_times[-1]
        first_cost_time = self.disc_cost_times[0]
        b = t.shape[0]
        S = self.num_s_per_t
        assert z_s.shape == (b, S, *self.shape_x)
        assert s_times.shape == mask_s.shape == (b, S, 1)

        minus_ones = (-1,) * len(self.shape_x)
        x0 = x0.unsqueeze(1).expand(-1, S, *minus_ones).reshape(b * S, *self.shape_x)
        x1 = x1.unsqueeze(1).expand(-1, S, *minus_ones).reshape(b * S, *self.shape_x)

        if fb == 'f':
            # sampling on (s_max, T)
            delta_t = self.T - last_cost_time
            s_boundary = last_cost_time + self.eps + (delta_t - 2 * self.eps) * torch.rand((b, S, 1),
                                                                                           device=t.device)
            # we always have s_boundary > s_times
        else:
            # sampling on (0, s_min)
            delta_t = first_cost_time
            s_boundary = self.eps + (delta_t - 2 * self.eps) * torch.rand((b, S, 1), device=t.device)
            # we always have s_boundary < s_times
        # s_boundary:(b,S,1)
        # z_s_boundary|z_s
        with torch.no_grad():
            z_s_boundary = gpath.sample_s_given_t(s_times.squeeze(-1),
                                                  z_s,
                                                  s_boundary,
                                                  fb).detach().squeeze(2)  # (B,S,dim)
        assert z_s_boundary.shape == (b, S, *self.shape_x)

        z_s_boundary = z_s_boundary.reshape(b * S, *self.shape_x)
        s_boundary = s_boundary.reshape(b * S, 1)
        mask_s = mask_s.reshape(b * S, 1)
        t_ = t.detach().clone().unsqueeze(1).expand(-1, S, -1).reshape(b * S, 1)

        # compute the missing CV term
        _, B_cv, _ = self.cv_net[fb](t_, s_boundary, fb=fb, target_s=s_boundary)
        if fb == 'f':
            target_boundary = -B_cv * delta_t * (x1 - z_s_boundary)
        else:
            target_boundary = B_cv * delta_t * (x0 - z_s_boundary)

        target_boundary *= mask_s
        assert target_boundary.shape == (b * S, *self.shape_x)
        return target_boundary
