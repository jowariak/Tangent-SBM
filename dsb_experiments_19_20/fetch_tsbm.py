"""Freeze authors' TSBM source on first setup; verify it on every later use."""
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parent/'vendor'/'tsbm'
REPO='maxencenoble/twisted-sb-matching'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def source_record():
    record=json.loads((ROOT/'manifest.json').read_text())
    if record['repository']!=REPO or not re.fullmatch('[0-9a-f]{40}',record['revision']):
        raise RuntimeError('Invalid TSBM source manifest')
    for name,h in record['files'].items():
        if sha(ROOT/name)!=h:raise RuntimeError(f'Modified TSBM source: {name}')
    return record

def download(url):
    req=urllib.request.Request(url,headers={'User-Agent':'DSB-experiment-source-fetcher'})
    with urllib.request.urlopen(req,timeout=90) as r:return r.read()

def main():
    if (ROOT/'manifest.json').exists():
        r=source_record();print('Verified frozen TSBM revision',r['revision'],flush=True);return
    ROOT.mkdir(parents=True,exist_ok=True)
    # Resolve main once; every file thereafter comes from this exact immutable SHA.
    # Re-running setup never silently updates an established installation.
    revision=json.loads(download(f'https://api.github.com/repos/{REPO}/commits/main'))['sha']
    if not re.fullmatch('[0-9a-f]{40}',revision):raise RuntimeError('Invalid GitHub revision')
    print('Freezing official TSBM revision',revision,flush=True)
    blob=download(f'https://codeload.github.com/{REPO}/tar.gz/{revision}')
    hashes={}
    with tarfile.open(fileobj=io.BytesIO(blob),mode='r:gz') as archive:
        for member in archive.getmembers():
            if not member.isfile():continue
            parts=PurePosixPath(member.name).parts[1:]
            if not parts or any(p in ['..','.'] for p in parts):raise RuntimeError('Unsafe archive path')
            name='/'.join(parts)
            if not (name.startswith('bridge/spline/') or name in ['bridge/sde/diffusion_bridge.py',
                    'bridge/trainer_tsbm.py','README.md','LICENSE','environment.yml']):continue
            path=ROOT.joinpath(*parts);path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(archive.extractfile(member).read());hashes[name]=sha(path)
    required=['bridge/spline/gaussian_path.py','bridge/spline/sde.py','bridge/sde/diffusion_bridge.py']
    if not all(p in hashes for p in required):raise RuntimeError('Upstream layout changed')
    record=dict(repository=REPO,revision=revision,archive_sha256=hashlib.sha256(blob).hexdigest(),files=hashes)
    (ROOT/'manifest.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
    print('Original source retained; SHA and file hashes saved.',flush=True)

if __name__=='__main__':main()
