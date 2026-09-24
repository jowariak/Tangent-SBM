#!/usr/bin/env python3
"""Extend completed MC1024 accumulators without modifying the original run."""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import torch
import evaluate as e


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('runs/lowdim_all_metrics_mc1024'))
    parser.add_argument('--out', type=Path, default=Path('runs/lowdim_all_metrics_mc4096'))
    parser.add_argument('--datasets', nargs='+', choices=list(e.DEFAULTS), default=list(e.DEFAULTS))
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    if args.source.resolve() == args.out.resolve():
        parser.error('Source and destination must differ; originals are preserved.')
    prefixes = [8,16,32,64,128,256,512,1024,2048,4096]
    here = Path(__file__).resolve().parent
    spec = json.loads(args.manifest.read_text()) if args.manifest else e.DEFAULTS
    commands, copies = [], []
    for dataset in args.datasets:
        module = e.gaussian if dataset == 'gaussian' else e.double_well
        command = [sys.executable, '-u', str(here/'evaluate.py'), '--dataset', dataset,
                   '--out', str(args.out), '--mc', *map(str,prefixes)]
        if args.manifest:
            command += ['--manifest', str(args.manifest)]
        subprocess.run(command + ['--check-only'], check=True)
        commands.append(command)
        # Fixed ID extension of exactly the original methods and three seeds.
        for method, template in spec[dataset]['models'].items():
            for seed in [32,42,52]:
                cp = Path(template.format(seed=seed))
                for kind in ['endpoint','sensitivity','finite_change']:
                    relative = Path(dataset)/method/f'seed_{seed}'/f'test_id_{kind}.pt'
                    source = args.source/relative
                    if not source.is_file():
                        raise FileNotFoundError(f'Missing saved accumulator: {source}')
                    saved = module.safe_torch_load(source)
                    prov = saved['provenance']
                    if saved['count'] != 1024 or prov['mc'] != prefixes[:8]:
                        raise ValueError(f'Expected completed MC1024 prefixes: {source}')
                    expected = dict(dataset=dataset,method=method,seed=seed,split='test_id',quantity=kind,
                        checkpoint=str(cp.resolve()),checkpoint_sha256=e.sha(cp),
                        data_sha256=e.sha(Path(spec[dataset]['data'])/'test_id.pt'),
                        metadata_sha256=e.sha(Path(spec[dataset]['data'])/'metadata.json'),
                        evaluator_sha256=e.sha(here/'evaluate.py'),base_sha256=e.sha(Path(module.__file__)),
                        torch_version=str(torch.__version__),draw_batch=8,condition_batch=256,
                        device='cuda' if torch.cuda.is_available() else 'cpu')
                    for key,value in expected.items():
                        if prov.get(key) != value:
                            raise ValueError(f'{source}: {key} changed; cannot safely reuse samples.')
                    if [r['mc'] for r in saved['records']] != prefixes[:8]:
                        raise ValueError(f'Incomplete prefix records: {source}')
                    if not all(torch.isfinite(saved[k]).all() for k in ['total','squares']):
                        raise ValueError(f'Nonfinite accumulator: {source}')
                    destination = args.out/relative
                    updated = copy.deepcopy(prov)
                    updated['mc'] = prefixes
                    if destination.exists():
                        existing = module.safe_torch_load(destination)
                        if existing['provenance'] != updated or not 1024 <= existing['count'] <= 4096:
                            raise ValueError(f'Incompatible extension already exists: {destination}')
                        continue
                    saved['provenance'] = updated
                    copies.append((destination,saved))
    # Validate everything before writing or running the extension.
    for destination,saved in copies:
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary = destination.with_suffix('.tmp')
        torch.save(saved,temporary)
        temporary.replace(destination)
    print(f'Prepared {len(copies)} accumulators. Existing extension progress is preserved.',flush=True)
    for command in commands:
        subprocess.run(command,check=True)


if __name__ == '__main__':
    main()
