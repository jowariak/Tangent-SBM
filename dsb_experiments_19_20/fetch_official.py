
import hashlib
import json
from pathlib import Path
import urllib.request

REVISION='a8ab5b500dcea8b1c0df84822188f69758a08442'
FILES=['gsbm/gaussian_path.py','gsbm/interp1d.py','gsbm/match_loss.py',
       'gsbm/sde.py','gsbm/ema.py','gsbm/pl_model.py','README.md','LICENSE.md']
ROOT=Path(__file__).resolve().parent/'vendor'/REVISION


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    manifest=ROOT/'manifest.json'
    if manifest.exists():
        record=json.loads(manifest.read_text())
        if record['revision']!=REVISION:raise RuntimeError('Wrong official source revision')
        for name,h in record['files'].items():
            if sha(ROOT/name)!=h:raise RuntimeError(f'Modified official source: {name}')
        print('Pinned official source verified.',flush=True);return
    hashes={}
    for name in FILES:
        path=ROOT/name;path.parent.mkdir(parents=True,exist_ok=True)
        url=f'https://raw.githubusercontent.com/facebookresearch/generalized-schrodinger-bridge-matching/{REVISION}/{name}'
        print('Downloading',name,flush=True)
        with urllib.request.urlopen(url,timeout=60) as r:data=r.read()
        path.write_bytes(data);hashes[name]=sha(path)
    
    manifest.write_text(json.dumps(dict(revision=REVISION,files=hashes),indent=2))
    print('Official source downloaded; original license retained.',flush=True)


if __name__=='__main__':main()
