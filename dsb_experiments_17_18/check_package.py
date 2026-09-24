"""Check the extracted bundle before importing training dependencies."""
import hashlib
import json
from pathlib import Path
import sys

root=Path(__file__).resolve().parent
manifest=root/'SHA256.json'
if not manifest.is_file():
    sys.exit('Incomplete experiment bundle: missing SHA256.json. Extract the full ZIP into /workspace.')
failures=[]
for name,expected in json.loads(manifest.read_text(encoding='utf-8')).items():
    path=root/name
    if not path.is_file():failures.append(f'Missing: {name}')
    elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:failures.append(f'Changed: {name}')
if failures:
    sys.exit('Experiment package check failed:\n'+'\n'.join(failures)+
             '\nExtract the complete supplied ZIP into /workspace, preserving its directory structure.')
print('PASS: complete experiment bundle and file hashes.',flush=True)
