"""Collect retrained models into the layout code/e080_infer.py reads, without touching the delivered weights.

weights: from <runs>/{ep24,e034,D1,D2}_fold{0,1,2} (outputs of the three training scripts) take the same files the delivered model uses
  (Mask R-CNN: last detector and last refiner checkpoint named in e006_receipt.json; dense: e004_fold{f}_unet.pt). Every run must be
  complete and match the settings in train/provenance.json; a smoke run or an early stop is refused. The files are copied to
  weights_retrained/ under the delivered names, write weights_retrained.sha256 and params_retrained.json (params.json with these paths;
  C2 and tau still the delivered ones until the next step).
c2: put the C2 parameters and tau of train/fit_c2.py --params params_retrained.json --out <file> into params_retrained.json.
Run from the repository root:
  python train/collect.py weights --runs train_runs
  python train/collect.py c2 --fit c2_retrained.json"""
import argparse, hashlib, json, shutil
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
OUT = 'weights_retrained'


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def weights(runs):
    par = json.loads((ROOT / 'params.json').read_text(encoding='utf-8'))
    prov = {m['id']: m['settings'] for m in json.loads((ROOT / 'train/provenance.json').read_text(encoding='utf-8'))['models']}
    (ROOT / OUT).mkdir(exist_ok=True)
    lines = []

    def put(src, rel):
        name = Path(rel).name
        dst = ROOT / OUT / name
        shutil.copyfile(src, dst)
        lines.append(f'{sha(dst)}  {OUT}/{name}')
        return f'{OUT}/{name}'
    for mm in par['models']:
        fam, f = mm['id'].split('_fold')
        run = Path(runs) / f'{fam}_fold{f}'
        if mm['kind'] == 'rcnn':
            rc = json.loads((run / 'e006_receipt.json').read_text(encoding='utf-8'))
            assert int(rc['fold']) == int(f) and rc.get('input_mode', 'gray') == mm['input_mode'], (run, rc.get('input_mode'))
            want, cfg, ck = prov[mm['id']], rc['config'], rc['checkpoints']
            assert rc['status'] == 'complete' and rc['terminal'], (run, 'run not complete')
            for k in ('seed', 'image_size', 'det_epochs', 'ref_epochs', 'train_records'):
                assert cfg[k] == want[k], (run, k, cfg[k], want[k])  # refuses smoke runs and other settings
            assert ck['det']['last']['epoch'] == want['det_epochs'] and ck['ref']['last']['epoch'] == want['ref_epochs'], (run, 'stopped early')
            mm['weights'] = put(run / ck['det']['last']['path'], mm['weights'])
            mm['refiner'] = put(run / ck['ref']['last']['path'], mm['refiner'])
        else:
            rc = json.loads((run / 'e004_receipt.json').read_text(encoding='utf-8'))
            want, cfg = prov[mm['id']], rc['config']
            assert int(rc['fold']) == int(f) and rc['terminal'] and rc['stopped_early'] is None, (run, 'run not complete')
            for k in ('size', 'epochs', 'batch', 'lr', 'encoder'):
                assert cfg[k] == want[k], (run, k, cfg[k], want[k])
            assert cfg.get('seed', 20261027) == want['seed'] and rc['epochs_completed'] == want['epochs'], (run, 'seed or epochs')
            mm['weights'] = put(run / f'e004_fold{f}_unet.pt', mm['weights'])
    (ROOT / f'{OUT}.sha256').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    par['weights_sha256'] = f'{OUT}.sha256'
    par['name'] = par['name'] + ' (retrained)'
    (ROOT / 'params_retrained.json').write_text(json.dumps(par, indent=1), encoding='utf-8')
    print(len(lines), 'checkpoints ->', OUT, '; params_retrained.json written (C2 and tau not yet refitted)')


def c2(fit):
    par = json.loads((ROOT / 'params_retrained.json').read_text(encoding='utf-8'))
    res = json.loads(Path(fit).read_text(encoding='utf-8'))
    assert not res.get('limit'), 'a --limit fit is a smoke test, not usable'
    par['c2'], par['tau'] = res['c2'], res['tau']
    (ROOT / 'params_retrained.json').write_text(json.dumps(par, indent=1), encoding='utf-8')
    print('params_retrained.json: C2 and tau from', fit, '| tau', res['tau'])


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('cmd', choices=['weights', 'c2']); ap.add_argument('--runs', default='train_runs'); ap.add_argument('--fit')
    a = ap.parse_args()
    weights(a.runs) if a.cmd == 'weights' else c2(a.fit)
