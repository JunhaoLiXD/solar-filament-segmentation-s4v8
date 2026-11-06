"""Instance fusion of same-fold members (all members held out on the evaluated fold).

Per member: v2 candidate pool (code/s1_candidates.pool: mask 0.5, score-ordered NMS 0.5, refiner 0.5, NMS) at
score >= 0.4, with that member's own refiner.
Fusion per image:
  1. all members' candidates sorted by score; greedy clustering: a candidate joins the first cluster whose anchor
     (highest-score member) has mask IoU > MATCH with it and that has no candidate from the same member yet;
  2. fused score = sum of member scores / M (missing member = 0);
  3. fused mask: 'anchor' = anchor mask; 'vote' = pixels covered by >= half of the cluster's members (fallback anchor);
  4. keep clusters with fused score >= t, then score-ordered NMS 0.5 with pixel subtraction (non-overlapping output).
Evaluation: exact pq_metric; t chosen on one half of the fold's groups and assessed on the other half, both ways,
summed (no selection on assessed images). Single members get the same treatment as comparators.

Run: ../../.venv-gpu/Scripts/python.exe code/f1_fusion.py --fold 0 --members e006 e010 e009
Out: preds/f1_fusion/fold{F}_{members}.json (+ cached member pools in preds/f1_fusion/pools/)
"""
import argparse
import itertools
import json
import pickle
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
import s1_candidates as s1  # noqa: E402
from a3_postproc import area, subtract  # noqa: E402

MEMBERS = {  # name -> run dir (fold 0 unless noted)
    'e006': 'kernels/e006-maskrcnn-refiner-fold0/output/v0',
    'e010': 'runs/b0-e006-seed2027-fold0',
    'e012': 'runs/b1-random-records-seed2027-fold0',
    'e009': 'kernels/e009-maskrcnn-1536-fold0/output/v0',
    'e015': 'kernels/e015-maskrcnn-1536-fold0-seed2027/output/v0',
    # folds 1-2 (1536): seed 2026 = E011, seed 2027 = E017
    'e011': {1: 'kernels/e011-maskrcnn-1536-fold1/output/v0', 2: 'kernels/e011-maskrcnn-1536-fold2/output/v0'},
    'e017': {1: 'kernels/e017-maskrcnn-1536-fold1-seed2027/output/v0',
             2: 'kernels/e017-maskrcnn-1536-fold2-seed2027/output/v0'},
}
MEMBERS['e025'] = 'kernels/e025-mask2former-swint-fold0/output/v0'
REFINER_FROM = {'e025': 'e015'}  # members without their own refiner
MEMBERS['e025q'] = MEMBERS['e025']  # count-matched scores (code/q1_quantile_calib.py), pool cache only
MEMBERS['e027'] = {1: 'kernels/e027-mask2former-swint-fold1/output/v0', 2: 'kernels/e027-mask2former-swint-fold2/output/v0'}
REFINER_FROM['e027'] = 'e017'  # Mask2Former folds 1-2 use the seed-2027 detector's refiner of the same fold
MEMBERS['e027q'] = MEMBERS['e027']
MEMBERS['e020'] = 'kernels/e020-maskrcnn-1536-ep24-fold0/output/v0'
MEMBERS['e024'] = {1: 'kernels/e024-maskrcnn-1536-ep24-fold1/output/v0', 2: 'kernels/e024-maskrcnn-1536-ep24-fold2/output/v0'}
MEMBERS['e034'] = {0: 'kernels/e034-maskrcnn-1536-phys-fold0/output/v0', 1: 'kernels/e034-maskrcnn-1536-phys-fold1/output/v0', 2: 'kernels/e034-maskrcnn-1536-phys-fold2/output/v0'}  # E034 phys input channels (fold 0)
# flip-TTA views (code/t1_tta.py): same run dir as the base member, pools cached under the view name
MEMBERS['e039'] = {k: f'runs/e039-s2026-ep24-fold{k}' for k in (0, 1, 2)}  # E039 seed-2026 24-epoch fold models (local)
MEMBERS['e041'] = {k: f'runs/e041-s2026-ep40-fold{k}' for k in (0, 1, 2)}  # E041 (C4) 40-epoch schedule, local
REFINER_FROM['e041'] = 'e039'  # C4: detector schedule is the only change vs E039
# E070 (hard-instance expert vs uniform control, Kaggle, replacing the ep24 slot): both use the ep24 slot's own refiner
MEMBERS['ep24slot'] = {0: MEMBERS['e020'], 1: MEMBERS['e024'][1], 2: MEMBERS['e024'][2]}
MEMBERS['e070u'] = {k: f'kernels/e070-u-fold{k}/output/v0' for k in (0, 1, 2)}
MEMBERS['e070x'] = {k: f'kernels/e070-x-fold{k}/output/v0' for k in (0, 1, 2)}
REFINER_FROM['e070u'] = REFINER_FROM['e070x'] = 'ep24slot'
for _b in ('e015', 'e017', 'e009', 'e011', 'e020', 'e024', 'e034', 'e039', 'e041', 'e070u', 'e070x'):
    for _v in ('hf', 'vf', 'hvf', 'tr', 'trhf', 'trvf', 'trhvf'):
        MEMBERS[_b + _v] = MEMBERS[_b]
    for _s in (1280, 1792):  # multi-scale TTA views
        for _v in ('', 'hf', 'vf', 'hvf', 'tr', 'trhf', 'trvf', 'trhvf'):
            MEMBERS[f'{_b}s{_s}{_v}'] = MEMBERS[_b]


def member_dir(name, fold):
    d = MEMBERS[name]
    return WORKDIR / (d[fold] if isinstance(d, dict) else d)
MATCH = 0.5
TS = np.round(np.arange(0.30, 0.96, 0.025), 3)
DATA = WORKDIR / 'data'


def member_pools(name, fold, ids, paths):
    cache_path = WORKDIR / f'preds/f1_fusion/pools/{name}_fold{fold}.pkl'
    if cache_path.exists():
        return pickle.loads(cache_path.read_bytes())
    kout = member_dir(name, fold)
    m_ids = json.loads((kout / f'e006_fold{fold}_oof_ids.json').read_text(encoding='utf-8'))
    assert m_ids == ids, f'{name}: OOF ids differ'
    rec = json.loads((kout / 'e006_receipt.json').read_text(encoding='utf-8'))
    device = torch.device('cuda')
    if name in REFINER_FROM:
        rdir = member_dir(REFINER_FROM[name], fold)
        rrec = json.loads((rdir / 'e006_receipt.json').read_text(encoding='utf-8'))
        ref, _ = e6.load_refiner(rdir / rrec['checkpoints']['ref']['last']['path'], device)
    else:
        ref, _ = e6.load_refiner(kout / rec['checkpoints']['ref']['last']['path'], device)
    raw = dict(np.load(kout / f'e006_fold{fold}_oof_det_last.npz'))
    factor = e6.Fraction(e6.FULL, int(raw['size']))
    per = e6.raw_by_image(raw, len(ids))
    pools = []
    for i, pid in enumerate(ids):
        cache = e6.RefinerCache(ref, device, e6.load_gray(paths[pid]))
        pools.append([(c, pl['score']) for c, pl in s1.pool(per[i], cache, 0.4, factor)])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_bytes(pickle.dumps(pools, protocol=4))
    return pools


def iou(a, b):
    i = e6._pair_intersection(a, b)
    return i / max(1, area(a) + area(b) - i)


def vote_mask(members):
    y0 = min(c[0] for c in members); x0 = min(c[1] for c in members)
    y1 = max(c[0] + c[2].shape[0] for c in members); x1 = max(c[1] + c[2].shape[1] for c in members)
    acc = np.zeros((y1 - y0, x1 - x0), np.uint16)  # uint8 overflowed in acc * 2 for > 127 members (found in E054, 128 members)
    for cy, cx, m in members:
        acc[cy - y0:cy - y0 + m.shape[0], cx - x0:cx - x0 + m.shape[1]] += m
    return e6.trim(y0, x0, acc * 2 >= len(members))


def fuse(member_lists, M):
    """member_lists: list (per member) of [(crop, score)]. -> clusters [(fused_score, anchor, members_crops)]."""
    allc = sorted([(s, k, c) for k, lst in enumerate(member_lists) for c, s in lst], key=lambda x: -x[0])
    clusters = []  # [anchor_crop, {member: (crop, score)}]
    for s, k, c in allc:
        for cl in clusters:
            if k not in cl[1] and iou(cl[0], c) > MATCH:
                cl[1][k] = (c, s)
                break
        else:
            clusters.append([c, {k: (c, s)}])
    return [(sum(s for _, s in cl[1].values()) / M, cl[0], [c for c, _ in cl[1].values()]) for cl in clusters]


def finalize(clusters, t, mode):
    kept = sorted([x for x in clusters if x[0] >= t], key=lambda x: -x[0])
    out = []
    for fs, anchor, mem in kept:
        c = anchor if mode == 'anchor' or len(mem) == 1 else (vote_mask(mem) or anchor)
        a = area(c)
        if not a or any(e6._pair_intersection(c, k) / max(1, min(a, area(k))) >= 0.5 for k in out):
            continue
        r = subtract(c, out)
        if r is not None:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--members', nargs='+', required=True)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--combos', nargs='*', default=None, help='only these fused sets (a+b+c), vote mode; plus single of the first member')
    ap.add_argument('--out-tag', default='')
    args = ap.parse_args()
    t0 = time.time()
    kout0 = member_dir(args.members[0], args.fold)
    ids = json.loads((kout0 / f'e006_fold{args.fold}_oof_ids.json').read_text(encoding='utf-8'))
    ann_path, images = e6.discover(DATA)
    paths = {s: p for s, (split, p) in images.items()}
    rec_phys, rec_anns = e6.load_annotations(ann_path, images)
    phys_recs = defaultdict(list)
    for rid, pid in sorted(rec_phys.items()):
        if rid in rec_anns:
            phys_recs[pid].append(rid)
    tasks = [(paths[p], None, [[a['segmentation'] for a in rec_anns[r]] for r in phys_recs[p]]) for p in ids]
    entries = [x[3] for x in e6.run_pool(e6.prep_task, tasks, args.workers)]
    pools = {m: member_pools(m, args.fold, ids, paths) for m in args.members}
    print(f'[{time.time()-t0:.0f}s] pools ready', flush=True)
    rng = np.random.default_rng(0)
    half = np.zeros(len(ids), int)
    half[rng.permutation(len(ids))[:len(ids) // 2]] = 1  # one physical image = one group in folds.csv

    configs = [('single', m, 'anchor') for m in args.members]
    if args.combos is not None:
        configs = [('single', args.members[0], 'anchor')] + [('fused', tuple(c.split('+')), 'vote') for c in args.combos]
    for r in ([] if args.combos is not None else range(2, len(args.members) + 1)):
        for combo in itertools.combinations(args.members, r):
            for mode in ('anchor', 'vote'):
                configs.append(('fused', combo, mode))
    results = {}
    for kind, combo, mode in configs:
        combo = (combo,) if kind == 'single' else combo
        M = len(combo)
        per_t = {t: [[], []] for t in TS}  # matrices per half
        for i in range(len(ids)):
            cl = fuse([pools[m][i] for m in combo], M)
            for t in TS:
                preds = finalize(cl, t, mode)
                for gts in entries[i]:
                    per_t[t][half[i]].append(e6.overlap(gts, preds))
        # choose t on one half, assess on the other, both ways; sum TP/FP/FN/IoU
        tot = {'tp': 0, 'fp': 0, 'fn': 0, 'tp_iou_sum': 0.0}
        chosen = []
        for sel, ass in ((0, 1), (1, 0)):
            t_best = max(TS, key=lambda t: pq.score_matrices(per_t[t][sel])['pq'])
            chosen.append(float(t_best))
            r_ = pq.score_matrices(per_t[t_best][ass])
            for k in tot:
                tot[k] += r_[k]
        pq_split = tot['tp_iou_sum'] / (tot['tp'] + 0.5 * tot['fp'] + 0.5 * tot['fn'])
        full_r = {float(t): pq.score_matrices(per_t[t][0] + per_t[t][1]) for t in TS}
        full = {t: r_['pq'] for t, r_ in full_r.items()}
        key = f'{kind}:{"+".join(combo)}:{mode}'
        t08 = full[min(full, key=lambda t: abs(t - 0.8))]
        results[key] = {'pq_split_selected': pq_split, 'thresholds_chosen': chosen,
                        'pq_at_0.8': t08, 'pq_best_full_biased': max(full.values()),
                        't_best_full': max(full, key=full.get),
                        'per_t': {str(t): {k: r_[k] for k in ('tp', 'fp', 'fn', 'tp_iou_sum')} for t, r_ in full_r.items()}}
        print(f'[{time.time()-t0:.0f}s] {key:40s} split {pq_split:.4f}  t{chosen}  @0.8 {t08:.4f}  best {max(full.values()):.4f}', flush=True)
    out = WORKDIR / f'preds/f1_fusion/fold{args.fold}_{args.out_tag or "_".join(args.members)}.json'
    out.write_text(json.dumps(results, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
