"""Print completed MC4096 comparisons without PyTorch or model checkpoints."""
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1]
for path in sorted((root/'results').glob('*.json')):
    print('\n'+path.stem)
    print('Method | Endpoint RMSE | E_J | Finite change RMSE')
    for row in json.loads(path.read_text())['aggregate']:
        if row['mc']!=4096: continue
        values=['{mean:.6f} +/- {sample_sd:.6f}'.format(**row[k]) for k in ['mean_rmse','E_J','finite_response_rmse']]
        print(' | '.join([row['method'],*values]))
