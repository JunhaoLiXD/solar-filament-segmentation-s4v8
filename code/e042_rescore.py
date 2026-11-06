"""E042 = C2 (pre-registered 2026-10-29, decisions.md): IoU-contribution rescoring of the A5 ensemble.
Fixed-geometry candidate pool (A5 clusters finalised at floor 0.30), filter only. Nested inside each fold's frozen
group halves: the assessed half never contributes to fitting or threshold choice. Arms: C2 (rescored), T0 (same pool,
fused-score threshold, same nested selection), A5 (E029 counts at the halves-chosen threshold).
Run: ../../.venv-gpu/Scripts/python.exe code/e042_rescore.py
Out: preds/e042/result.json, preds/e042/fold{f}_cands.pkl (candidate features / targets cache)"""
import json, pickle, sys
from pathlib import Path
import numpy as np
from scipy import ndimage
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
import e006_maskrcnn_refiner as e6  # noqa: E402
import f1_fusion as f1  # noqa: E402
import g1_group_eval as g1  # noqa: E402
import pq_metric as pq  # noqa: E402
from a3_postproc import area, subtract  # noqa: E402
from e029_fusion_structure import MODELS, VIEWS  # noqa: E402
from s2_rescore import fit_predict  # noqa: E402

FAMS = ('s2026', 's2027', 'ep24')  # A5 member order (same as e039_eval / e029)
M = len(FAMS) * len(VIEWS)
FLOOR = 0.30
TAU_C2 = np.round(np.arange(0.02, 0.805, 0.01), 3)
TAU_T0 = np.round(np.arange(0.30, 0.951, 0.025), 3)
KIN = 5
OUT = W / 'preds/e042'


def fuse_k(member_lists):
    """f1.fuse, keeping member index and score of every cluster member."""
    allc = sorted([(s, k, c) for k, lst in enumerate(member_lists) for c, s in lst], key=lambda x: -x[0])
    clusters = []
    for s, k, c in allc:
        for cl in clusters:
            if k not in cl[1] and f1.iou(cl[0], c) > f1.MATCH:
                cl[1][k] = (c, s)
                break
        else:
            clusters.append([c, {k: (c, s)}])
    return [(sum(s for _, s in cl[1].values()) / M, cl[0], cl[1]) for cl in clusters]


def finalize_idx(clusters, t):
    """f1.finalize(..., 'vote') returning (mask, cluster index)."""
    order = sorted([j for j, x in enumerate(clusters) if x[0] >= t], key=lambda j: -clusters[j][0])
    out, idx = [], []
    for j in order:
        fs, anchor, mem = clusters[j]
        crops = [c for c, _ in mem.values()]
        c = anchor if len(crops) == 1 else (f1.vote_mask(crops) or anchor)
        a = area(c)
        if not a or any(e6._pair_intersection(c, k) / max(1, min(a, area(k))) >= 0.5 for k in out):
            continue
        r = subtract(c, out)
        if r is not None:
            out.append(r); idx.append(j)
    return out, idx


def contrast(img, c):
    y, x, m = c
    p = 6
    y0, x0 = max(0, y - p), max(0, x - p)
    y1, x1 = min(img.shape[0], y + m.shape[0] + p), min(img.shape[1], x + m.shape[1] + p)
    big = np.zeros((y1 - y0, x1 - x0), bool)
    big[y - y0:y - y0 + m.shape[0], x - x0:x - x0 + m.shape[1]] = m
    ring = ndimage.binary_dilation(big, iterations=5) & ~big
    sub = img[y0:y1, x0:x1].astype(np.float32)
    bg = sub[ring].mean() if ring.any() else sub.mean()
    return float((bg - sub[big].mean()) / max(bg, 1.0))


def build_fold(f):
    cache = OUT / f'fold{f}_cands.pkl'
    if cache.exists():
        return pickle.loads(cache.read_bytes())
    ids, entries, groups, half, paths = g1.fold_data(f)
    names = [MODELS[m][f] + v for m in FAMS for v in VIEWS]
    pools = [f1.member_pools(n, f, ids, paths) for n in names]
    imgs = []
    for i, pid in enumerate(ids):
        cls = fuse_k([p[i] for p in pools])
        masks, idx = finalize_idx(cls, FLOOR)
        assert [(a[0], a[1], a[2].tobytes()) for a in masks] == \
               [(a[0], a[1], a[2].tobytes()) for a in f1.finalize([(x[0], x[1], [c for c, _ in x[2].values()]) for x in cls], FLOOR, 'vote')]
        gray = e6.load_gray(paths[pid])
        R = len(entries[i])
        dS = np.zeros(len(masks)); dD = np.zeros(len(masks)); hitrec = np.zeros(len(masks))
        for g in entries[i]:
            iou = e6.overlap(g, masks) if len(masks) else np.zeros((len(g), 0))
            hit = iou > 0.5
            k = hit.sum(0) if len(g) else np.zeros(len(masks), int)
            dS += np.where(hit, iou, 0).sum(0) if len(g) else 0
            dD += np.where(k > 0, 0.5 * k, 0.5)
            hitrec += k > 0
        X = []
        for c, j in zip(masks, idx):
            fs, _, mem = cls[j]
            ks = list(mem)
            sc = [s for _, s in mem.values()]
            yy, xx = np.nonzero(c[2])
            cy, cx = c[0] + yy.mean(), c[1] + xx.mean()
            X.append([fs, len(ks) / M, float(np.mean(sc)), len({k // len(VIEWS) for k in ks}) / len(FAMS),
                      len({k % len(VIEWS) for k in ks}) / len(VIEWS), np.log1p(area(c)), contrast(gray, c),
                      float(np.hypot(cy - 1024, cx - 1024) / 1024), np.log1p(len(masks))])
        imgs.append({'masks': masks, 'X': np.array(X, np.float64).reshape(-1, 9), 'dS': dS, 'dD': dD,
                     'u': np.clip(dS / R, 0, 1), 'y': hitrec / R, 'gt': sum(len(g) for g in entries[i])})
        if i % 50 == 0:
            print(f'fold {f} image {i}/{len(ids)}', flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    cache.write_bytes(pickle.dumps(imgs, protocol=4))
    return imgs


def additive_pq(imgs, sel, keep_fn):
    s = d = gt = 0.0
    for i in sel:
        k = keep_fn(i)
        s += imgs[i]['dS'][k].sum(); d += imgs[i]['dD'][k].sum(); gt += imgs[i]['gt']
    return s / (0.5 * gt + d) if (0.5 * gt + d) > 0 else 0.0


def best_tau(imgs, sel, score, taus):
    vals = [additive_pq(imgs, sel, lambda i: score[i] >= t) for t in taus]
    return float(taus[int(np.argmax(vals))])


def stack(imgs, sel, key):
    return np.concatenate([imgs[i][key] for i in sel]) if sel else np.zeros(0)


def fit_on(imgs, sel, target):
    X = np.concatenate([imgs[i]['X'] for i in sel]); y = stack(imgs, sel, target)
    return X, y


def predict_split(imgs, fit_sel, pred_sel, target):
    X, y = fit_on(imgs, fit_sel, target)
    Xe = np.concatenate([imgs[i]['X'] for i in pred_sel])
    p = fit_predict('logreg', X.astype(np.float32), y, np.ones(len(y)), Xe.astype(np.float32))
    out, o = {}, 0
    for i in pred_sel:
        n = len(imgs[i]['dS']); out[i] = p[o:o + n]; o += n
    return out


def nested_fold(imgs, groups, half, target):
    """-> per-image keep masks for C2 and T0, chosen taus."""
    keep_c2, keep_t0, taus = {}, {}, []
    for s, a in ((0, 1), (1, 0)):
        S = [i for i in range(len(imgs)) if half[i] == s]; A = [i for i in range(len(imgs)) if half[i] == a]
        ug = sorted({groups[i] for i in S})
        perm = np.random.default_rng(0).permutation(len(ug))
        part = {g: int(perm[k] % KIN) for k, g in enumerate(ug)}
        cross = {}
        for q in range(KIN):
            ins = [i for i in S if part[groups[i]] == q]; trn = [i for i in S if part[groups[i]] != q]
            cross.update(predict_split(imgs, trn, ins, target))
        tau = best_tau(imgs, S, cross, TAU_C2)
        pa = predict_split(imgs, S, A, target)
        for i in A:
            keep_c2[i] = pa[i] >= tau
        fsc = {i: imgs[i]['X'][:, 0] for i in range(len(imgs))}
        t0 = best_tau(imgs, S, fsc, TAU_T0)
        for i in A:
            keep_t0[i] = fsc[i] >= t0
        taus.append({'sel_half': s, 'tau_c2': tau, 'tau_t0': t0})
    return keep_c2, keep_t0, taus


def official_counts(imgs, keep, entries):
    C = np.zeros((len(imgs), 4))
    for i, ent in enumerate(entries):
        preds = [m for m, k in zip(imgs[i]['masks'], keep[i]) if k]
        r = pq.score_matrices([e6.overlap(g, preds) for g in ent])
        C[i] = (r['tp'], r['fp'], r['fn'], r['tp_iou_sum'])
    return C


def a5_counts(f, half):
    C = np.load(W / 'preds/e029/arm_A5.npz')[f'f{f}']
    out = np.zeros((len(C), 4))
    for s, a in ((0, 1), (1, 0)):
        k = int(np.argmax(g1.pq_of(C[half == s].sum(0))))
        out[half == a] = C[half == a, k]
    return out


def bootstrap(arms, groups_by_fold, n=1000, seed=0):
    """arms: name -> {fold: (n_img, 4)}. Groups resampled within folds; thresholds fixed (nested choice)."""
    rng = np.random.default_rng(seed)
    folds = sorted(groups_by_fold)
    prep = {}
    for f in folds:
        ug = sorted(set(groups_by_fold[f])); gi = {g: k for k, g in enumerate(ug)}
        gidx = np.array([gi[g] for g in groups_by_fold[f]])
        prep[f] = {a: np.zeros((len(ug), 4)) for a in arms}
        for a in arms:
            np.add.at(prep[f][a], gidx, arms[a][f])
    names = list(arms)
    reps = {a: np.zeros(n) for a in names}
    for r in range(n):
        tot = {a: np.zeros(4) for a in names}
        for f in folds:
            w = rng.multinomial(len(prep[f][names[0]]), np.ones(len(prep[f][names[0]])) / len(prep[f][names[0]])).astype(float)
            for a in names:
                tot[a] += (w[:, None] * prep[f][a]).sum(0)
        for a in names:
            reps[a][r] = g1.pq_of(tot[a])
    return reps


def compare(arms, reps, a, b):
    pa = g1.pq_of(sum(arms[a][f].sum(0) for f in arms[a])); pb = g1.pq_of(sum(arms[b][f].sum(0) for f in arms[b]))
    d = reps[b] - reps[a]
    pf = {f: float(g1.pq_of(arms[b][f].sum(0)) - g1.pq_of(arms[a][f].sum(0))) for f in arms[a]}
    res = {'pq_a': float(pa), 'pq_b': float(pb), 'delta': float(pb - pa), 'p_delta_gt0': float((d > 0).mean()),
           'ci90': [float(np.percentile(d, 5)), float(np.percentile(d, 95))], 'per_fold_delta': pf,
           'folds_positive': int(sum(v > 0 for v in pf.values()))}
    res['accept_free'] = bool(res['delta'] >= 0.0015 and res['p_delta_gt0'] >= 0.9 and res['folds_positive'] >= 2)
    return res


def main():
    arms = {'A5': {}, 'T0': {}, 'C2': {}, 'C2y': {}}
    groups_by_fold, taus = {}, {}
    for f in (0, 1, 2):
        ids, entries, groups, half, paths = g1.fold_data(f)
        imgs = build_fold(f)
        groups_by_fold[f] = list(groups)
        arms['A5'][f] = a5_counts(f, half)
        kc, kt, tu = nested_fold(imgs, groups, half, 'u')
        arms['C2'][f] = official_counts(imgs, kc, entries)
        arms['T0'][f] = official_counts(imgs, kt, entries)
        ky, _, tuy = nested_fold(imgs, groups, half, 'y')  # secondary diagnostic (agreement target)
        arms['C2y'][f] = official_counts(imgs, ky, entries)
        taus[f] = {'u': tu, 'y': tuy}
        print(f'fold {f}: ' + ' '.join(f'{a} {g1.pq_of(arms[a][f].sum(0)):.4f}' for a in arms), flush=True)
    a5 = g1.pq_of(sum(arms['A5'][f].sum(0) for f in arms['A5']))
    z = np.load(W / 'preds/e029/arm_A5.npz'); ref = g1.evaluate({f: z[f'f{f}'] for f in (0, 1, 2)})['pq']
    assert abs(a5 - ref) < 1e-9, f'A5 reconstruction {a5} vs {ref}'
    reps = bootstrap(arms, groups_by_fold)
    res = {'C2_vs_T0': compare(arms, reps, 'T0', 'C2'), 'C2_vs_A5': compare(arms, reps, 'A5', 'C2'),
           'T0_vs_A5': compare(arms, reps, 'A5', 'T0'), 'C2y_vs_A5 (diagnostic)': compare(arms, reps, 'A5', 'C2y'),
           'taus': taus, 'adopt': None,
           'note': 'bootstrap without per-replicate re-selection (selection nested with fitting); reused development folds'}
    res['adopt'] = bool(res['C2_vs_T0']['accept_free'] and res['C2_vs_A5']['accept_free'])
    for k in ('C2_vs_T0', 'C2_vs_A5', 'T0_vs_A5', 'C2y_vs_A5 (diagnostic)'):
        r = res[k]
        print(f"{k}: {r['pq_a']:.4f} -> {r['pq_b']:.4f} d {r['delta']:+.4f} P {r['p_delta_gt0']:.2f} "
              f"ci90 {[round(x, 4) for x in r['ci90']]} folds {[round(v, 4) for v in r['per_fold_delta'].values()]} "
              f"accept {r['accept_free']}", flush=True)
    print('adopt', res['adopt'])
    (OUT / 'result.json').write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')


if __name__ == '__main__':
    main()
