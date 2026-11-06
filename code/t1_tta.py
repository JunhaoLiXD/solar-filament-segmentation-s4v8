"""Flip TTA for a Mask R-CNN member: run the detector on the h-flipped / v-flipped held-out images (local GPU),
map every instance back to the original orientation, then build the v2 candidate pool with the member's own
refiner on the ORIGINAL image (code/s1_candidates.pool). Pools are cached where code/f1_fusion.py reads them
(preds/f1_fusion/pools/{member}{hf|vf}_fold{F}.pkl), so the TTA views are fused like extra members:
  ../../.venv-gpu/Scripts/python.exe code/f1_fusion.py --fold F --members e015 e015hf e015vf

Run: ../../.venv-gpu/Scripts/python.exe code/t1_tta.py --member e015 --fold 0
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
W = HERE.parent
sys.path.insert(0, str(HERE))
import e006_maskrcnn_refiner as e6  # noqa: E402
import s1_candidates as s1  # noqa: E402
import f1_fusion as f1  # noqa: E402


VIEWS = {  # name -> (flip axes applied first, transpose after); inverse = transpose back, then same flips
    'id': ((), False),
    'hf': ((1,), False), 'vf': ((0,), False), 'hvf': ((0, 1), False),
    'tr': ((), True), 'trhf': ((1,), True), 'trvf': ((0,), True), 'trhvf': ((0, 1), True)}


def view_raw(model, pids, paths, device, size, view):
    """Like e6.detector_raw on a transformed input; per-image instance lists in ORIGINAL orientation."""
    flips, tr = VIEWS[view]
    per = []
    with torch.inference_mode():
        for pid in pids:
            img = e6.resize_input(e6.load_gray(paths[pid]), size)
            if flips:
                img = np.flip(img, axis=flips)
            if tr:
                img = img.T
            x = torch.from_numpy(e6.det_input(img)).to(device)  # grey x 3, or E034 phys channels (e6.INPUT_MODE)
            out = model([x])[0]
            keep = out['scores'] >= min(e6.GRID_SCORE)
            pr = out['masks'][keep, 0]
            if tr:
                pr = pr.transpose(1, 2)
            if flips:
                pr = torch.flip(pr, dims=[a + 1 for a in flips])
            lv = (pr >= 0.4).to(torch.uint8) + (pr >= 0.5).to(torch.uint8) + (pr >= 0.6).to(torch.uint8)
            nz = lv > 0
            rows, cols = nz.any(2).cpu().numpy(), nz.any(1).cpu().numpy()
            lv = lv.cpu().numpy()
            insts = []
            for k, sc in enumerate(out['scores'][keep].cpu().numpy()):
                r, c = np.flatnonzero(rows[k]), np.flatnonzero(cols[k])
                if not len(r):
                    continue
                y0, y1, x0, x1 = int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1
                insts.append((float(sc), y0, x0, lv[k, y0:y1, x0:x1].copy()))
            per.append(insts)
    return per


def flipped_raw(model, pids, paths, device, size, axis):
    return view_raw(model, pids, paths, device, size, 'hf' if axis == 1 else 'vf')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--member', required=True)
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--views', nargs='+', default=['hf', 'vf'], choices=list(VIEWS))
    ap.add_argument('--size', type=int, default=None, help='detector input size override (multi-scale TTA)')
    a = ap.parse_args()
    t0 = time.time()
    device = torch.device('cuda')
    kout = f1.member_dir(a.member, a.fold)
    ids = json.loads((kout / f'e006_fold{a.fold}_oof_ids.json').read_text(encoding='utf-8'))
    ann_path, images = e6.discover(W / 'data')
    paths = {s: p for s, (split, p) in images.items()}
    rec = json.loads((kout / 'e006_receipt.json').read_text(encoding='utf-8'))
    e6.INPUT_MODE = rec.get('input_mode', 'gray')
    size = a.size or int(np.load(kout / f'e006_fold{a.fold}_oof_det_last.npz')['size'])
    prefix = a.member + (f's{a.size}' if a.size else '')
    det, _ = e6.load_detector(kout / rec['checkpoints']['det']['last']['path'], device, size)
    if a.member in f1.REFINER_FROM:  # same rule as f1_fusion.member_pools
        rdir = f1.member_dir(f1.REFINER_FROM[a.member], a.fold)
        rrec = json.loads((rdir / 'e006_receipt.json').read_text(encoding='utf-8'))
        ref, _ = e6.load_refiner(rdir / rrec['checkpoints']['ref']['last']['path'], device)
    else:
        ref, _ = e6.load_refiner(kout / rec['checkpoints']['ref']['last']['path'], device)
    factor = e6.Fraction(e6.FULL, size)
    # sanity: identity view through the same code path vs Kaggle raw on 3 images (counts only)
    raw = e6.raw_by_image(dict(np.load(kout / f'e006_fold{a.fold}_oof_det_last.npz')), len(ids))
    for tag in a.views:
        cache_path = W / f'preds/f1_fusion/pools/{prefix}{tag if tag != "id" else ""}_fold{a.fold}.pkl'
        if cache_path.exists():
            print('cached', cache_path); continue
        per = view_raw(det, ids, paths, device, size, tag)
        print(f'[{time.time()-t0:.0f}s] {tag}: {sum(map(len, per))} instances (original view {sum(map(len, raw))})', flush=True)
        pools = []
        for i, pid in enumerate(ids):
            cache = e6.RefinerCache(ref, device, e6.load_gray(paths[pid]))
            pools.append([(c, pl['score']) for c, pl in s1.pool(per[i], cache, 0.4, factor)])
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_bytes(pickle.dumps(pools, protocol=4))
        print(f'[{time.time()-t0:.0f}s] -> {cache_path}', flush=True)


if __name__ == '__main__':
    main()
