"""E029 fusion structure (decisions.md 2026-10-28, pre-registered candidates A0-A5).
Run: ../../.venv-gpu/Scripts/python.exe code/e029_fusion_structure.py
Out: preds/e029/result.json, preds/e029/arm_*.npz (count arrays for g1_group_eval)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
W = HERE.parent
sys.path.insert(0, str(HERE))
import e006_maskrcnn_refiner as e6  # noqa: E402
import f1_fusion as f1  # noqa: E402
import g1_group_eval as g1  # noqa: E402
from a3_postproc import area, subtract  # noqa: E402

VIEWS = ('', 'hf', 'vf', 'hvf', 'tr', 'trhf', 'trvf', 'trhvf')
MODELS = {'s2027': {0: 'e015', 1: 'e017', 2: 'e017'}, 's2026': {0: 'e009', 1: 'e011', 2: 'e011'},
          'ep24': {0: 'e020', 1: 'e024', 2: 'e024'}}
ARMS = {'A0': ('flat', ['s2027']), 'A1': ('flat', ['s2026', 's2027']), 'A2': ('two', ['s2026', 's2027']),
        'A3': ('flat', ['s2027', 'ep24']), 'A4': ('two', ['s2026', 's2027', 'ep24']), 'A5': ('flat', ['s2026', 's2027', 'ep24'])}
LEVEL1_FLOOR = 0.3


def level1(view_lists):
    """One model's 8 views -> non-overlapping instances [(vote crop, mean score)] with score >= floor, score order."""
    cl = sorted([x for x in f1.fuse(view_lists, len(view_lists)) if x[0] >= LEVEL1_FLOOR], key=lambda x: -x[0])
    out = []
    for fs, anchor, mem in cl:
        c = anchor if len(mem) == 1 else (f1.vote_mask(mem) or anchor)
        a = area(c)
        if not a or any(e6._pair_intersection(c, k) / max(1, min(a, area(k))) >= 0.5 for k, _ in out):
            continue
        r = subtract(c, [k for k, _ in out])
        if r is not None:
            out.append((r, fs))
    return out


def main():
    out_dir = W / 'preds/e029'
    out_dir.mkdir(parents=True, exist_ok=True)
    arms = {a: {} for a in ARMS}
    for fold in (0, 1, 2):
        ids, entries, groups, half, paths = g1.fold_data(fold)
        pools = {m: [f1.member_pools(MODELS[m][fold] + v, fold, ids, paths) for v in VIEWS] for m in MODELS}
        for name, (kind, models) in ARMS.items():
            if kind == 'flat':
                lists = lambda i: [p[i] for m in models for p in pools[m]]  # noqa: E731
                n = len(models) * len(VIEWS)
            else:
                l1 = {m: [level1([p[i] for p in pools[m]]) for i in range(len(ids))] for m in models}
                lists = lambda i: [l1[m][i] for m in models]  # noqa: E731
                n = len(models)
            cls = [f1.fuse(lists(i), n) for i in range(len(ids))]
            arms[name][fold] = g1.counts(lambda i, t: f1.finalize(cls[i], t, 'vote'), entries)
            print(f'fold {fold} {name} done', flush=True)
    for name in arms:
        np.savez(out_dir / f'arm_{name}.npz', **{f'f{f}': arms[name][f] for f in arms[name]})
    res = {'arms': {}, 'vs_A0': {}}
    for name in arms:
        ev = g1.evaluate(arms[name])
        res['arms'][name] = {'pq': ev['pq'], 'pq_rotation_old': ev['pq_rotation_old'],
                             'per_fold': {f: v['pq_halves'] for f, v in ev['per_fold'].items()},
                             'thresholds': {f: v['thresholds'] for f, v in ev['per_fold'].items()}}
    # inner joint selection of (arm, threshold) on one half, applied to the other half
    tot, picks = np.zeros(4), []
    for fold in (0, 1, 2):
        half = g1.fold_data(fold)[3]
        for sel, ass in ((0, 1), (1, 0)):
            best = max(((name, k) for name in arms for k in range(len(g1.TS))),
                       key=lambda nk: g1.pq_of(arms[nk[0]][fold][half == sel].sum(0)[nk[1]]))
            picks.append(best[0])
            tot += arms[best[0]][fold][half == ass].sum(0)[best[1]]
    res['inner_selection'] = {'pq': float(g1.pq_of(tot)), 'picks': picks,
                              'pick_counts': {a: picks.count(a) for a in arms}}
    top = max(arms, key=lambda a: (picks.count(a), res['arms'][a]['pq']))
    res['adopt_candidate'] = top
    for name in arms:
        if name != 'A0':
            b = g1.paired_bootstrap(arms['A0'], arms[name])
            b['accept_free_rule'] = g1.accept(b, 'free')
            res['vs_A0'][name] = b
    (out_dir / 'result.json').write_text(json.dumps(res, indent=1), encoding='utf-8')
    for name, v in res['arms'].items():
        extra = res['vs_A0'].get(name, {})
        print(f"{name} pq {v['pq']:.4f} rot {v['pq_rotation_old']:.4f} folds {[round(x, 4) for x in v['per_fold'].values()]}"
              + (f" d {extra['delta']:+.4f} P {extra['p_delta_gt0']:.2f} ci90 {[round(x, 4) for x in extra['ci90']]} "
                 f"f+ {extra['folds_positive']} accept {extra['accept_free_rule']}" if extra else ''))
    print('inner selection', res['inner_selection'], 'candidate', top)


if __name__ == '__main__':
    main()
