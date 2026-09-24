"""Dependency-free source and numeric-record checks; not a training test."""
import ast
import hashlib
import json
from pathlib import Path
import statistics

root=Path(__file__).resolve().parents[1]
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
count=0
for path in root.rglob('*.py'):
    if '.venv' in path.parts: continue
    ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
    count+=1
for entry in json.loads((root/'SOURCE_MANIFEST.json').read_text()):
    assert sha(root/entry['path'])==entry['sha256'],entry['path']
for manifest in root.glob('*/vendor/**/manifest.json'):
    record=json.loads(manifest.read_text())
    for name,digest in record.get('files',{}).items():
        if not isinstance(digest,str): continue
        assert sha(manifest.parent/name)==digest,(str(manifest),name)
for path in (root/'results').glob('*.json'):
    data=json.loads(path.read_text())
    for row in data['aggregate']:
        for key in ['mean_rmse','E_J','finite_response_rmse']:
            values=[r[key] for item in data['per_seed']
                if item['provenance']['method']==row['method'] and item['provenance']['split']==row['split']
                for r in item['results'] if r['mc']==row['mc'] and key in r]
            assert len(values)==3
            assert abs(statistics.mean(values)-row[key]['mean'])<1e-12
            assert abs(statistics.stdev(values)-row[key]['sample_sd'])<1e-12
print(f'PASS: {count} Python files parsed; source manifests and result aggregates verified.')
