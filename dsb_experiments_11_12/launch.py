"""Run #11/#12 inside the user's normal Compose environment.

No image building or GPU selection. Install ipdb in this same container first.
"""
import argparse
from pathlib import Path
import subprocess
import sys

HERE=Path(__file__).resolve().parent

def run(script,*args):
    subprocess.run([sys.executable,'-u',str(HERE/script),*args],check=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['setup','smoke','train','evaluate','all'])
    p.add_argument('experiment',choices=['11','12'])
    a,extra=p.parse_known_args()
    script='run_pde_tsbm.py' if a.experiment=='11' else 'run_sns_gsbm.py'
    if a.mode in ['setup','smoke','train','all']:
        run('fetch_official.py')
        if a.experiment=='11':run('fetch_tsbm.py')
        
        run(script,'smoke')
    if a.mode in ['train','all']:
        for seed in [32,42,52]:run(script,'train','--seed',str(seed),*extra)
    if a.mode in ['evaluate','all']:run(script,'evaluate',*extra)

if __name__=='__main__':main()
