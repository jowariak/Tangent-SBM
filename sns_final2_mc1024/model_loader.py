"""Frozen-model, validation-only response Monte Carlo convergence diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import spdebench_sns_conditional_dsbm_v2 as base
from spdebench_sns_eval_final2 import response_diag
from frozen_common import audit_tangent

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def write(p, obj):
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    tmp.replace(p)

def checkpoint(args, method, seed):
    roots = dict(conditional=args.conditional_root, tangent=args.tangent_root,
                 gsbm=args.gsbm_root, tsbm=args.tsbm_root)
    root = roots[method]
    if method == 'conditional': root = root/'sigma_1p4'
    if method == 'tangent': root = root/'lambda_1000'
    return root/f'seed_{seed}'/('imf_5.pt' if method in ('conditional','tangent') else 'final.pt')

def model(args, method, seed, meta, device):
    cp = checkpoint(args, method, seed)
    ck = base.safe_load(cp)
    cfg = base.Config(**ck['config'])
    mean, std = base.normalization_from_metadata(meta)
    assert cfg.seed == seed and cfg.reference_sigma == 1.4
    m = base.ConditionalFieldDSBM(cfg, device)
    if method in ('conditional', 'tangent'):
        assert ck['imf'] == 5
        assert abs(ck['state_mean']-mean)<1e-8 and abs(ck['state_std']-std)<1e-8
        if method == 'tangent': audit_tangent(ck)
        m.load_state_dict(ck['model'])
    else:
        assert ck['normalization'] == meta['normalization']
        assert ck['train_data_sha256'] == sha(args.data_dir/'endpoint_train.pt')
        assert ck['completed_passes'] == (10 if method == 'tsbm' else 2*ck['options']['cycles'])
        m.net_f.load_state_dict(ck['weights']['fwd'])
        m.net_b.load_state_dict(ck['weights']['bwd'])
    m.net_f.eval(); m.net_b.eval()
    return m, mean, std
