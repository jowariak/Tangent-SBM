"""Optional #10 startup compatibility fix; never called by the #11/#12 runner."""
import argparse
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--script',type=Path,default=Path('pdebench_gsbm_official/run_pde_gsbm.py'))
    p.add_argument('--run-root',type=Path,default=Path('runs/pdebench_gsbm_official_v1'))
    a=p.parse_args()
    text=a.script.read_text(encoding='utf-8')
    old="net=self.nets[direction];net.train();trace=[]"
    new="net=self.nets[direction];net.train(True);trace=[]"
    if old not in text and new in text:print('EMA call is already compatible.');return
    if text.count(old)!=1:raise RuntimeError('Unexpected #10 adapter; refusing automatic change')
    if any(a.run_root.glob('seed_*/*.pt')):
        raise RuntimeError('Existing checkpoints: source hashes must be preserved; do not patch a completed/running experiment')
    backup=a.script.with_suffix('.before_ema_fix.py')
    with backup.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    text=text.replace(old,new).replace('directory.mkdir(parents=True);history=[]',
                                      'directory.mkdir(parents=True,exist_ok=True);history=[]')
    with a.script.open('w',encoding='utf-8',newline='\n') as f:f.write(text)
    print('Fixed explicit EMA train mode; backup:',backup)

if __name__=='__main__':main()
