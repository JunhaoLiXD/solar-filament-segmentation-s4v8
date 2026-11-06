"""P0.1 shared evaluation (plan v3 section 1; Codex review #1-#3).

An "arm" = per fold, a count array C[f] of shape (n_images, n_thresholds, 4) with (tp, fp, fn, tp_iou_sum) summed over
the image's annotator records (official PQ semantics, pq_metric.score_matrices).
- primary: within each fold, the frozen groups (folds.csv group_id) are split into two halves (fixed seed); the threshold
  is chosen on one half and applied to the other, both ways; counts summed over halves and folds -> PQ.
- old: fold rotation (threshold for fold k chosen on the other folds' summed counts) — reported separately.
- paired group bootstrap: groups resampled with replacement within each fold (same draw for both arms); thresholds
  re-selected inside every replicate with the same half procedure; delta of summed PQ.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE.parent
sys.path.insert(0, str(HERE))
import e006_maskrcnn_refiner as e6  # noqa: E402
import pq_metric as pq  # noqa: E402

TS = np.round(np.arange(0.30, 0.96, 0.025), 3)
FOLD_BASE = {0: 'e015', 1: 'e017', 2: 'e017'}


def pq_of(c):
    d = c[..., 0] + 0.5 * c[..., 1] + 0.5 * c[..., 2]
    return np.divide(c[..., 3], d, out=np.zeros_like(d, dtype=float), where=d > 0)


_DATA = {}


def fold_data(fold, workers=6):
    """-> ids, entries (GT crops per record), group ids, half labels (0/1 per image)."""
    if fold in _DATA:
        return _DATA[fold]
    import f1_fusion as f1
    base = FOLD_BASE[fold]
    ids = json.loads((f1.member_dir(base, fold) / f'e006_fold{fold}_oof_ids.json').read_text(encoding='utf-8'))
    ann_path, images = e6.discover(W / 'data')
    paths = {s: p for s, (split, p) in images.items()}
    rec_phys, rec_anns = e6.load_annotations(ann_path, images)
    phys = defaultdict(list)
    for rid, pid in sorted(rec_phys.items()):
        if rid in rec_anns:
            phys[pid].append(rid)
    tasks = [(paths[p], None, [[a['segmentation'] for a in rec_anns[r]] for r in phys[p]]) for p in ids]
    entries = [x[3] for x in e6.run_pool(e6.prep_task, tasks, workers)]
    grp = {}
    with open(W / 'folds.csv', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            grp[r['physical_id']] = r['group_id']
    groups = [grp[p] for p in ids]
    ug = sorted(set(groups))
    rng = np.random.default_rng(1000 + fold)
    perm = rng.permutation(len(ug))
    half_of = {g: int(k < len(ug) // 2) for g, k in zip(ug, np.argsort(perm))}
    half = np.array([half_of[g] for g in groups])
    _DATA[fold] = (ids, entries, groups, half, paths)
    return _DATA[fold]


def counts(preds_at, entries, ts=TS):
    """preds_at(i, t) -> list of crops for image i at threshold t. -> (n, T, 4)."""
    C = np.zeros((len(entries), len(ts), 4))
    for i, ent in enumerate(entries):
        for k, t in enumerate(ts):
            preds = preds_at(i, t)
            r = pq.score_matrices([e6.overlap(g, preds) for g in ent])
            C[i, k] = (r['tp'], r['fp'], r['fn'], r['tp_iou_sum'])
    return C


def pool_counts(pool, entries, ts=TS):
    """pool = per image [(crop, score)] already score-ordered NMS'ed (s1 pool / fused) -> counts."""
    return counts(lambda i, t: [c for c, s in pool[i] if s >= t], entries, ts)


def _halves_select(Cg_by_half, ts=TS):
    """Cg_by_half: (2, T, 4) summed counts per half -> summed counts after cross-half selection, chosen ts."""
    tot, chosen = np.zeros(4), []
    for sel, ass in ((0, 1), (1, 0)):
        k = int(np.argmax(pq_of(Cg_by_half[sel])))
        chosen.append(float(ts[k]))
        tot += Cg_by_half[ass, k]
    return tot, chosen


def evaluate(arm, ts=TS):
    """arm: {fold: C}. -> dict with primary (group halves) and old (fold rotation) PQ, per fold."""
    out = {'per_fold': {}}
    tot = np.zeros(4)
    for f, C in arm.items():
        half = fold_data(f)[3]
        H = np.stack([C[half == 0].sum(0), C[half == 1].sum(0)])
        c, ch = _halves_select(H, ts)
        tot += c
        out['per_fold'][f] = {'pq_halves': float(pq_of(c)), 'thresholds': ch, 'pq_best_biased': float(pq_of(C.sum(0)).max()),
                              't_best_biased': float(ts[int(np.argmax(pq_of(C.sum(0))))])}
    out['pq'] = float(pq_of(tot))
    out['counts'] = tot.tolist()
    rot = np.zeros(4)
    for f, C in arm.items():
        others = sum(arm[g].sum(0) for g in arm if g != f)
        if isinstance(others, np.ndarray):
            k = int(np.argmax(pq_of(others)))
            rot += C.sum(0)[k]
            out['per_fold'][f]['t_rotation'] = float(ts[k])
    out['pq_rotation_old'] = float(pq_of(rot)) if len(arm) > 1 else None
    return out


def paired_bootstrap(arm_a, arm_b, n=1000, seed=0, ts=TS, margin=None):
    """Delta = PQ(b) - PQ(a), groups resampled within folds, thresholds re-selected per replicate."""
    rng = np.random.default_rng(seed)
    folds = sorted(arm_a)
    prep = {}
    for f in folds:
        ids, entries, groups, half, _ = fold_data(f)
        ug = sorted(set(groups))
        gi = {g: k for k, g in enumerate(ug)}
        gidx = np.array([gi[g] for g in groups])
        Ga = np.zeros((len(ug), len(ts), 4)); Gb = np.zeros_like(Ga)
        np.add.at(Ga, gidx, arm_a[f]); np.add.at(Gb, gidx, arm_b[f])
        gh = np.array([half[gidx == k][0] for k in range(len(ug))])
        prep[f] = (Ga, Gb, gh)
    deltas = np.zeros(n)
    for r in range(n):
        ta, tb = np.zeros(4), np.zeros(4)
        for f in folds:
            Ga, Gb, gh = prep[f]
            w = rng.multinomial(len(gh), np.ones(len(gh)) / len(gh)).astype(float)
            for G, acc in ((Ga, ta), (Gb, tb)):
                H = np.stack([(w[gh == h, None, None] * G[gh == h]).sum(0) for h in (0, 1)])
                acc += _halves_select(H, ts)[0]
        deltas[r] = pq_of(tb) - pq_of(ta)
    ea, eb = evaluate(arm_a, ts), evaluate(arm_b, ts)
    return {'pq_a': ea['pq'], 'pq_b': eb['pq'], 'delta': eb['pq'] - ea['pq'],
            'p_delta_gt0': float((deltas > 0).mean()),
            **({'p_delta_gt_margin': float((deltas > margin).mean()), 'margin': margin} if margin is not None else {}), 'ci90': [float(np.percentile(deltas, 5)), float(np.percentile(deltas, 95))],
            'per_fold_delta': {f: eb['per_fold'][f]['pq_halves'] - ea['per_fold'][f]['pq_halves'] for f in folds},
            'folds_positive': int(sum(eb['per_fold'][f]['pq_halves'] > ea['per_fold'][f]['pq_halves'] for f in folds)),
            'rotation_old': [ea['pq_rotation_old'], eb['pq_rotation_old']]}


def accept(res, kind):
    """Plan v3 section 1 rule. kind: 'free' (post-processing / fusion) or 'train'."""
    need = 0.0015 if kind == 'free' else 0.003
    return bool(res['delta'] >= need and res['p_delta_gt0'] >= 0.9 and res['folds_positive'] >= 2)


if __name__ == '__main__':
    # self-check: baseline seed-2027 8-view TTA vote at the old rotation must reproduce 0.4411
    import f1_fusion as f1
    VIEWS = ('', 'hf', 'vf', 'hvf', 'tr', 'trhf', 'trvf', 'trhvf')
    arm = {}
    for f, base in FOLD_BASE.items():
        ids, entries, groups, half, paths = fold_data(f)
        pools = [f1.member_pools(base + v, f, ids, paths) for v in VIEWS]
        cls = [f1.fuse([p[i] for p in pools], len(VIEWS)) for i in range(len(ids))]
        arm[f] = counts(lambda i, t: f1.finalize(cls[i], t, 'vote'), entries)
        print(f, 'groups', len(set(groups)), 'half sizes', int((half == 0).sum()), int((half == 1).sum()), flush=True)
    ev = evaluate(arm)
    print(json.dumps(ev, indent=1))
    out = W / 'preds/g1_group_eval'
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / 'arm_A0_tta8_seed2027.npz', **{f'f{f}': arm[f] for f in arm})
    (out / 'selfcheck.json').write_text(json.dumps(ev, indent=1), encoding='utf-8')
