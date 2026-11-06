"""A3: zero-training post-processing ablations on cached E006 raw OOF (any fold).

Variants (all predictions kept non-overlapping: the metric counts every IoU>0.5 pair, so overlapping
predictions would be a scorer exploit):
  merge = 'transitive'  source rule: any shared pixel -> union (detector stage and refiner stage)
  merge = 'nms<t>'      score-ordered: drop a mask whose intersection / min(area) >= t with a kept mask,
                        otherwise keep it minus pixels already taken; same after refinement
  score threshold, refiner threshold, min_area (2048 px), optional E004 mean-probability gate (fold 0).

Run (from WORKDIR): ../../.venv-gpu/Scripts/python.exe code/a3_postproc.py --fold 0
Output: preds/a3_postproc/fold{F}/grid.csv (+ report.json)
"""
import argparse
import csv
import itertools
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKDIR = HERE.parent
sys.path.insert(0, str(HERE))
import e006_maskrcnn_refiner as e6  # noqa: E402
import pq_metric as pq  # noqa: E402
from a2_error_analysis import e006_dir  # noqa: E402

DATA = WORKDIR / 'data'
SCORES = (0.7, 0.8, 0.85, 0.9)
REFS = (0.5, 0.65)
MERGES = ('transitive', 'nms0.3', 'nms0.5')
MIN_AREAS = (0, 512, 2048)
E004_GATES = (0.0, 0.2, 0.3, 0.4)


def area(c):
    return int(c[2].sum())


def subtract(c, taken):
    """c minus pixels of the already-kept masks."""
    y0, x0, m = c
    m = m.copy()
    for ty, tx, tm in taken:
        ya, yb = max(y0, ty), min(y0 + m.shape[0], ty + tm.shape[0])
        xa, xb = max(x0, tx), min(x0 + m.shape[1], tx + tm.shape[1])
        if ya < yb and xa < xb:
            m[ya - y0:yb - y0, xa - x0:xb - x0] &= ~tm[ya - ty:yb - ty, xa - tx:xb - tx]
    return e6.trim(y0, x0, m)


def nms(crops, t):
    """crops sorted by score desc."""
    kept = []
    for c in crops:
        a = area(c)
        if not a:
            continue
        if any(e6._pair_intersection(c, k) / max(1, min(a, area(k))) >= t for k in kept):
            continue
        r = subtract(c, kept)
        if r is not None:
            kept.append(r)
    return kept


def predictions(insts, cache, score, ref, merge, factor):
    sel = sorted([x for x in insts if x[0] >= score and x[3].size], key=lambda x: -x[0])
    crops = [c for c in (e6.trim(y0, x0, lv >= 2) for s, y0, x0, lv in sel) if c is not None]  # mask 0.5
    crops = [e6.upscale(c, factor) for c in crops]
    if merge == 'transitive':
        crops = e6.merge_crops(crops)
        return e6.refined_masks(crops, cache, ref)[0] if crops else []
    t = float(merge[3:])
    crops = nms(crops, t)
    if not crops:
        return []
    out = []
    for c, ((x0, y0, x1, y1), p) in zip(crops, cache.probs(crops)):
        r = e6.trim(y0, x0, p >= ref)
        out.append(r if r is not None else c)
    return nms(out, t)  # order still = detector score order


def e004_mean(prob1024, c):
    y0, x0, m = c
    sub = prob1024[y0 // 2:(y0 + m.shape[0] + 1) // 2, x0 // 2:(x0 + m.shape[1] + 1) // 2].astype(np.float32)
    mm = m[::2, ::2][:sub.shape[0], :sub.shape[1]]
    return float(sub[:mm.shape[0], :mm.shape[1]][mm].mean()) if mm.any() else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--run-dir', default=None, help='E006-format output dir (default: E006/E008 Kaggle output)')
    ap.add_argument('--tag', default='', help='output subfolder suffix')
    args = ap.parse_args()
    t0 = time.time()
    out = WORKDIR / f'preds/a3_postproc/fold{args.fold}{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    kout = Path(args.run_dir) if args.run_dir else e006_dir(args.fold)
    ids = json.loads((kout / f'e006_fold{args.fold}_oof_ids.json').read_text(encoding='utf-8'))
    ann_path, images = e6.discover(DATA)
    paths = {s: p for s, (split, p) in images.items()}
    rec_phys, rec_anns = e6.load_annotations(ann_path, images)
    phys_recs = defaultdict(list)
    for rid, pid in sorted(rec_phys.items()):
        if rid in rec_anns:
            phys_recs[pid].append(rid)
    tasks = [(paths[p], None, [[a['segmentation'] for a in rec_anns[r]] for r in phys_recs[p]]) for p in ids]
    entries = {p: x[3] for p, x in zip(ids, e6.run_pool(e6.prep_task, tasks, args.workers))}
    device = torch.device('cuda')
    rec = json.loads((kout / 'e006_receipt.json').read_text(encoding='utf-8'))
    ref_model, _ = e6.load_refiner(kout / rec['checkpoints']['ref']['last']['path'], device)
    per = e6.raw_by_image(dict(np.load(kout / f'e006_fold{args.fold}_oof_det_last.npz')), len(ids))
    e4 = None
    if args.fold == 0:
        k4 = WORKDIR / 'kernels/e004-unet-fold0/output/v0'
        e4_ids = json.loads((k4 / 'e004_fold0_oof_ids.json').read_text(encoding='utf-8'))
        e4 = (np.load(k4 / 'e004_fold0_oof_probs.npy', mmap_mode='r'), {p: i for i, p in enumerate(e4_ids)})
    print(f'[{time.time()-t0:.0f}s] prepared {len(ids)} images', flush=True)

    base = list(itertools.product(SCORES, REFS, MERGES))
    gates = E004_GATES if e4 else (0.0,)
    mats = defaultdict(list)
    counts = defaultdict(int)
    factor = e6.Fraction(e6.FULL, int(np.load(kout / f"e006_fold{args.fold}_oof_det_last.npz")["size"]))
    for i, pid in enumerate(ids):
        cache = e6.RefinerCache(ref_model, device, e6.load_gray(paths[pid]))
        prob = np.asarray(e4[0][e4[1][pid]]) if e4 else None
        for (s, r, mg) in base:
            preds = predictions(per[i], cache, s, r, mg, factor)
            means = [e004_mean(prob, c) for c in preds] if prob is not None else None
            for ma in MIN_AREAS:
                for g in gates:
                    keep = [c for k, c in enumerate(preds) if area(c) >= ma and (not g or means[k] >= g)]
                    key = (s, r, mg, ma, g)
                    counts[key] += len(keep)
                    for gts in entries[pid]:
                        mats[key].append(e6.overlap(gts, keep))
        if i % 40 == 0:
            print(f'[{time.time()-t0:.0f}s] image {i+1}/{len(ids)}', flush=True)
    rows = []
    for key, ms in mats.items():
        res = pq.score_matrices(ms)
        rows.append({'score': key[0], 'refiner': key[1], 'merge': key[2], 'min_area': key[3], 'e004_gate': key[4],
                     'pq': res['pq'], 'tp': res['tp'], 'fp': res['fp'], 'fn': res['fn'], 'pred_instances': counts[key]})
    rows.sort(key=lambda x: -x['pq'])
    with open(out / 'grid.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    ref_row = next(x for x in rows if (x['score'], x['refiner'], x['merge'], x['min_area'], x['e004_gate']) == (0.8, 0.5, 'transitive', 0, 0.0))
    report = {'fold': args.fold, 'locked_baseline': ref_row, 'top10': rows[:10], 'seconds': time.time() - t0}
    (out / 'report.json').write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
