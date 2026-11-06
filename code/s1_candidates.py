"""Step 1 (agreement rescoring): candidate table with exact per-candidate metric contributions.

Pool = v2 pipeline (mask 0.5, score-ordered NMS 0.5, refiner 0.5, NMS again) run at a LOW score threshold
(default 0.4 = raw OOF floor). NMS is in score order, so the pool filtered at score >= t is exactly the v2 output
at threshold t (v2 = t 0.8). Final masks are non-overlapping, so a GT can be hit (IoU > 0.5) by at most one
candidate and contributions add up exactly:
    per record r, candidate c hitting k GTs: dS = sum IoU of the hits, dD = 0.5*k if k > 0 else 0.5
    PQ(subset) = sum dS / (0.5 * total GT + sum dD)
Removing a candidate keeps the others' geometry fixed (Codex review: fixed-geometry filtering).

Run: ../../.venv-gpu/Scripts/python.exe code/s1_candidates.py --fold 0 [--run-dir DIR --tag _x]
Out: preds/s1_candidates/fold{F}{tag}/candidates.csv, images.csv, check.json
"""
import argparse
import csv
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
from a2_error_analysis import e006_dir, disk_params  # noqa: E402
from a3_postproc import area, subtract  # noqa: E402

DATA = WORKDIR / 'data'
NMS_T, MASK_LEVEL, REF_T = 0.5, 2, 0.5


def nms_idx(items, t):
    """items: (crop, payload) sorted by score desc -> kept (crop minus taken pixels, payload)."""
    kept = []
    for c, pl in items:
        a = area(c)
        if not a:
            continue
        if any(e6._pair_intersection(c, k) / max(1, min(a, area(k))) >= t for k, _ in kept):
            continue
        r = subtract(c, [k for k, _ in kept])
        if r is not None:
            kept.append((r, pl))
    return kept


def pool(insts, cache, score, factor):
    sel = sorted([x for x in insts if x[0] >= score and x[3].size], key=lambda x: -x[0])
    items = []
    for s, y0, x0, lv in sel:
        c = e6.trim(y0, x0, lv >= MASK_LEVEL)
        if c is not None:
            items.append((e6.upscale(c, factor), {'score': s, 'det_mean_level': float(lv[lv > 0].mean())}))
    items = nms_idx(items, NMS_T)
    if not items:
        return []
    out = []
    for (c, pl), ((x0, y0, x1, y1), p) in zip(items, cache.probs([c for c, _ in items])):
        r = e6.trim(y0, x0, p >= REF_T)
        pl = dict(pl, refiner_fallback=int(r is None), refiner_max=float(p.max()))
        out.append((r if r is not None else c, pl, (x0, y0, p)))
    final = nms_idx([(c, (pl, pp)) for c, pl, pp in out], NMS_T)
    res = []
    for c, (pl, (x0, y0, p)) in final:
        cy0, cx0, m = c
        sub = p[cy0 - y0:cy0 - y0 + m.shape[0], cx0 - x0:cx0 - x0 + m.shape[1]]
        pl = dict(pl, refiner_mean=float(sub[m].mean()) if sub.shape == m.shape and m.any() else 0.0)
        res.append((c, pl))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--pool-score', type=float, default=0.4)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    t0 = time.time()
    out = WORKDIR / f'preds/s1_candidates/fold{args.fold}{args.tag}'
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
    raw = dict(np.load(kout / f'e006_fold{args.fold}_oof_det_last.npz'))
    factor = e6.Fraction(e6.FULL, int(raw['size']))
    per = e6.raw_by_image(raw, len(ids))
    groups = {}
    with open(WORKDIR / 'folds.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            groups[r['physical_id']] = r['group_id']
    print(f'[{time.time()-t0:.0f}s] prepared {len(ids)} images', flush=True)

    cand_rows, img_rows, mats_v2, multi_hits = [], [], [], 0
    for i, pid in enumerate(ids):
        cache = e6.RefinerCache(ref_model, device, e6.load_gray(paths[pid]))
        cands = pool(per[i], cache, args.pool_score, factor)
        cy, cx, R = disk_params(paths[pid])
        recs = entries[pid]
        n_gt = sum(len(g) for g in recs)
        img_rows.append({'image': pid, 'group': groups[pid], 'fold': args.fold, 'records': len(recs), 'n_gt': n_gt,
                         'n_cand': len(cands)})
        preds = [c for c, _ in cands]
        dS = np.zeros(len(cands)); dD = np.zeros(len(cands)); hits = np.zeros(len(cands), int)
        for gts in recs:
            iou = e6.overlap(gts, preds) if preds else np.zeros((len(gts), 0), np.float32)
            hit = iou > 0.5
            if len(gts) and len(preds):
                if (hit.sum(0) > 1).any():
                    multi_hits += int((hit.sum(0) > 1).sum())
                if (hit.sum(1) > 1).any():
                    raise RuntimeError(f'GT hit by two candidates in {pid}: non-overlap assumption broken')
            k = hit.sum(0) if len(gts) else np.zeros(len(preds), int)
            dS += np.where(hit, iou, 0).sum(0) if len(gts) else 0
            dD += np.where(k > 0, 0.5 * k, 0.5)
            hits += (k > 0)
            keep = [j for j, (_, pl) in enumerate(cands) if pl['score'] >= 0.8]
            mats_v2.append(iou[:, keep] if len(preds) else iou)
        for j, (c, pl) in enumerate(cands):
            y0, x0, m = c
            a = int(m.sum()); h, w = m.shape
            ys, xs = np.nonzero(m)
            rr = float(np.hypot(ys.mean() + y0 - cy, xs.mean() + x0 - cx) / R)
            cand_rows.append({'image': pid, 'group': groups[pid], 'fold': args.fold, 'cand': j, **pl,
                              'area': a, 'h': h, 'w': w, 'thickness': a / max(h, w), 'fill': a / (h * w),
                              'r_over_R': rr, 'records': len(recs), 'hits': int(hits[j]), 'dS': float(dS[j]),
                              'dD': float(dD[j]), 'y': hits[j] / len(recs), 'u': dS[j] / len(recs)})
        if i % 40 == 0:
            print(f'[{time.time()-t0:.0f}s] image {i+1}/{len(ids)} cands {len(cands)}', flush=True)

    for name, rows in (('candidates.csv', cand_rows), ('images.csv', img_rows)):
        with open(out / name, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    # checks: exact scorer on the v2 subset vs additive formula
    exact = pq.score_matrices(mats_v2)
    gt_total = sum(r['n_gt'] for r in img_rows)
    sel = [r for r in cand_rows if r['score'] >= 0.8]
    additive = sum(r['dS'] for r in sel) / (0.5 * gt_total + sum(r['dD'] for r in sel))
    check = {'fold': args.fold, 'run_dir': str(kout), 'pool_score': args.pool_score, 'candidates': len(cand_rows),
             'v2_exact': exact, 'v2_additive_pq': additive, 'gt_total': gt_total,
             'candidates_hitting_2plus_gt_in_one_record': multi_hits, 'seconds': time.time() - t0}
    (out / 'check.json').write_text(json.dumps(check, indent=1), encoding='utf-8')
    print(json.dumps(check, indent=1))


if __name__ == '__main__':
    main()
