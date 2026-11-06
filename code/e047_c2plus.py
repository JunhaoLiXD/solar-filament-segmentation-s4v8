"""E047 (pre-registered 2026-10-29, decisions.md + plan.md rev 2): richer descriptors (41, frozen schema) + LightGBM
scorer (cross_entropy on u, weight R) on E042's fixed-geometry pool. Arms: E042 (logreg 9, w=1), E042R (logreg 9, w=R),
LGB9R (LightGBM 9, w=R), E047 (LightGBM 41, w=R). Nested group halves x inner seeds {0,1,2}; paired bootstrap of the
seed-mean delta; LOFO transfer diagnostic (seed 0).
Run: ../../.venv-gpu/Scripts/python.exe code/e047_c2plus.py [--pilot]      Out: preds/e047/"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
OUT = W / 'preds/e047'
CAP_SECONDS = 4 * 3600
T0 = time.time()
UNET = {0: 'kernels/e004-unet-fold0/output/v0', 1: 'kernels/e022-unet-1024-fold1/output/v0', 2: 'kernels/e022-unet-1024-fold2/output/v0'}
SEEDS = (0, 1, 2)
ARMS = ('E042', 'E042R', 'LGB9R', 'E047')
LGB = dict(objective='cross_entropy', num_leaves=15, learning_rate=0.05, min_data_in_leaf=40, feature_fraction=0.8,
           bagging_fraction=0.8, bagging_freq=1, seed=0, deterministic=True, num_threads=4, verbose=-1)
NAMES = (['fused', 'vote', 'mean_present', 'fams', 'views', 'log_area', 'contrast', 'dist', 'log_ncand',
          'smax', 'smin', 'sstd', 'sq25', 'sq75', 'fam_mean_0', 'fam_mean_1', 'fam_mean_2', 'fam_vote_0', 'fam_vote_1',
          'fam_vote_2', 'fam_views_0', 'fam_views_1', 'fam_views_2', 'feret_length', 'area_over_feret', 'elongation',
          'solidity', 'bbox_aspect', 'abs_sin_theta', 'pair_iou', 'vote_over_union', 'kept_over_pre', 'nn_dist',
          'n_within_100', 'score_rank', 'n_fused_065', 'disk_mean', 'disk_std', 'unet_mean', 'unet_q90', 'unet_share05'])
assert len(NAMES) == 41


def check_cap(where):
    if time.time() - T0 > CAP_SECONDS:
        raise SystemExit(f'wall-clock cap {CAP_SECONDS} s exceeded at {where}')


# ---------------------------------------------------------------- features
def shape_feats(c):
    import cv2
    y0, x0, m = c
    area = int(m.sum())
    cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    pts = np.concatenate([q.reshape(-1, 2) for q in cnts]) if cnts else np.zeros((1, 2), np.int32)
    hull = cv2.convexHull(pts.astype(np.int32)).reshape(-1, 2)
    if len(hull) >= 2:
        d = hull[:, None, :].astype(np.float64) - hull[None, :, :]
        feret = float(np.sqrt((d ** 2).sum(-1)).max())
    else:
        feret = 0.0
    feret = max(feret, 1.0)
    if len(hull) >= 3:
        hm = np.zeros(m.shape, np.uint8)
        cv2.fillConvexPoly(hm, hull.astype(np.int32), 1)
        hull_px = int(hm.sum())
        solidity = area / max(hull_px, area) if hull_px else 1.0
    else:
        solidity = 1.0
    aof = area / feret
    elong = feret / max(aof, 1.0)
    bbox = max(m.shape) / max(min(m.shape), 1)
    yy, xx = np.nonzero(m)
    sin = 0.0
    if len(yy) > 1:
        cov = np.cov(np.stack([xx, yy]).astype(np.float64))
        ev, evec = np.linalg.eigh(cov)
        if abs(ev[1] - ev[0]) >= 1e-9:
            vx, vy = evec[:, 1]
            sin = float(abs(vy) / max(np.hypot(vx, vy), 1e-12))
    return [feret, aof, elong, solidity, bbox, sin]


def pair_iou(crops, bbox_seed):
    k = len(crops)
    if k < 2:
        return 1.0
    pairs = [(a, b) for a in range(k) for b in range(a + 1, k)]
    if len(pairs) > 276:
        rng = np.random.default_rng(bbox_seed)
        pairs = [pairs[j] for j in rng.choice(len(pairs), 276, replace=False)]
    return float(np.mean([f1.iou(crops[a], crops[b]) for a, b in pairs]))


def union_area(crops):
    y0 = min(c[0] for c in crops); x0 = min(c[1] for c in crops)
    y1 = max(c[0] + c[2].shape[0] for c in crops); x1 = max(c[1] + c[2].shape[1] for c in crops)
    acc = np.zeros((y1 - y0, x1 - x0), bool)
    for y, x, m in crops:
        acc[y - y0:y - y0 + m.shape[0], x - x0:x - x0 + m.shape[1]] |= m
    return int(acc.sum())


def finalize_pre(clusters, t):
    """r42.finalize_idx, also returning the pre-subtraction vote mask of every kept candidate."""
    order = sorted([j for j, x in enumerate(clusters) if x[0] >= t], key=lambda j: -clusters[j][0])
    out, idx, pre = [], [], []
    for j in order:
        fs, anchor, mem = clusters[j]
        crops = [c for c, _ in mem.values()]
        c = anchor if len(crops) == 1 else (f1.vote_mask(crops) or anchor)
        a = r42.area(c)
        if not a or any(e6._pair_intersection(c, k) / max(1, min(a, r42.area(k))) >= 0.5 for k in out):
            continue
        r = r42.subtract(c, out)
        if r is not None:
            out.append(r); idx.append(j); pre.append(c)
    return out, idx, pre


def unet_map(probs_row):
    import torch
    t = torch.from_numpy(np.asarray(probs_row, np.float32))[None, None]
    return torch.nn.functional.interpolate(t, size=(2048, 2048), mode='bilinear', align_corners=False)[0, 0].numpy()


def image_features(member_lists, M, gray, pmap):
    """-> masks (final), X (n, 41). Shared by OOF and test."""
    cls = r42.fuse_k(member_lists) if M == r42.M else fuse_any(member_lists, M)
    masks, idx, pre = finalize_pre(cls, r42.FLOOR)
    n = len(masks)
    yy_disk, xx_disk = np.ogrid[:2048, :2048]
    disk = (yy_disk - 1024) ** 2 + (xx_disk - 1024) ** 2 <= 900 ** 2
    dmean, dstd = float(gray[disk].mean() / 255), float(gray[disk].std() / 255)
    cents = []
    for c in masks:
        yy, xx = np.nonzero(c[2]); cents.append((c[0] + yy.mean(), c[1] + xx.mean()))
    cents = np.array(cents).reshape(-1, 2)
    fused_all = np.array([cls[j][0] for j in idx])
    n065 = int((fused_all >= 0.65).sum())
    per_fam = M // 3
    X = []
    for q, (c, j, pc) in enumerate(zip(masks, idx, pre)):
        fs, _, mem = cls[j]
        ks = list(mem); sc = np.array([s for _, s in mem.values()])
        crops = [cc for cc, _ in mem.values()]
        cy, cx = cents[q]
        base = [fs, len(ks) / M, float(sc.mean()), len({k // per_fam for k in ks}) / 3, len({k % 8 for k in ks}) / 8,
                np.log1p(r42.area(c)), r42.contrast(gray, c), float(np.hypot(cy - 1024, cx - 1024) / 1024), np.log1p(n)]
        stats = [float(sc.max()), float(sc.min()), float(sc.std()), float(np.quantile(sc, 0.25)), float(np.quantile(sc, 0.75))]
        fam = []
        for fi in range(3):
            fk = [k for k in ks if k // per_fam == fi]
            fam.append(sum(mem[k][1] for k in fk) / per_fam)
        famv = [len([k for k in ks if k // per_fam == fi]) / per_fam for fi in range(3)]
        famw = [len({k % 8 for k in ks if k // per_fam == fi}) / 8 for fi in range(3)]
        h, w = c[2].shape
        seed = (c[0] * 2048 + c[1]) * 4096 + h * 64 + w
        cons = [pair_iou(crops, seed), r42.area(pc) / max(union_area(crops), 1)]
        kept = [r42.area(c) / max(r42.area(pc), 1)]
        if n > 1:
            dd = np.hypot(cents[:, 0] - cy, cents[:, 1] - cx); dd[q] = np.inf
            nb = [float(dd.min() / 1024), int((dd <= 100).sum())]
        else:
            nb = [1.0, 0]
        rank = [float((fused_all > fs).sum() + 1) / n]
        img = [n065, dmean, dstd]
        sub = pmap[c[0]:c[0] + h, c[1]:c[1] + w][c[2]]
        un = [float(sub.mean()), float(np.quantile(sub, 0.9)), float((sub >= 0.5).mean())]
        X.append(base + stats + fam + famv + famw + shape_feats(c) + cons + kept + nb + rank + img + un)
    return masks, np.array(X, np.float64).reshape(-1, 41)


def fuse_any(member_lists, M):
    allc = sorted([(s, k, c) for k, lst in enumerate(member_lists) for c, s in lst], key=lambda x: -x[0])
    cl = []
    for s, k, c in allc:
        for x in cl:
            if k not in x[1] and f1.iou(x[0], c) > f1.MATCH:
                x[1][k] = (c, s); break
        else:
            cl.append([c, {k: (c, s)}])
    return [(sum(s for _, s in x[1].values()) / M, x[0], x[1]) for x in cl]


def build_fold(f, pilot=None):
    cache = OUT / f'fold{f}_feats.pkl'
    if cache.exists() and pilot is None:
        return pickle.loads(cache.read_bytes())
    import cv2  # noqa: F401
    from e029_fusion_structure import MODELS, VIEWS
    ids, entries, groups, half, paths = g1.fold_data(f, workers=2)
    names = [MODELS[m][f] + v for m in r42.FAMS for v in VIEWS]
    pools = [f1.member_pools(nm, f, ids, paths) for nm in names]
    old = pickle.loads((W / f'preds/e042/fold{f}_cands.pkl').read_bytes())
    ud = W / UNET[f]
    uids = json.loads((ud / f'e004_fold{f}_oof_ids.json').read_text(encoding='utf-8'))
    probs = np.load(ud / f'e004_fold{f}_oof_probs.npy', mmap_mode='r')
    urow = {p: k for k, p in enumerate(uids)}
    assert set(ids) <= set(urow), 'U-Net OOF ids do not cover the fold'
    out, tm = [], {'features': 0.0}
    rng_imgs = range(len(ids)) if pilot is None else range(pilot)
    for i in rng_imgs:
        t = time.time()
        gray = e6.load_gray(paths[ids[i]])
        masks, X = image_features([p[i] for p in pools], r42.M, gray, unet_map(probs[urow[ids[i]]]))
        assert len(masks) == len(old[i]['masks']) and all(
            (a[0], a[1], a[2].tobytes()) == (b[0], b[1], b[2].tobytes()) for a, b in zip(masks, old[i]['masks'])), f'pool mismatch image {i}'
        assert np.allclose(X[:, :9], old[i]['X'], atol=1e-9), f'E042 features mismatch image {i}'
        R = len(entries[i])
        out.append({'X': X, 'dS': old[i]['dS'], 'dD': old[i]['dD'], 'u': old[i]['u'], 'gt': old[i]['gt'], 'R': R})
        tm['features'] += time.time() - t
        if i % 25 == 0:
            print(f'[{time.time() - T0:.0f}s] fold {f} image {i}/{len(ids)}', flush=True)
            check_cap(f'features fold {f}')
    if pilot is None:
        OUT.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(pickle.dumps(out, protocol=4))
    return (out, tm) if pilot is not None else out


# ---------------------------------------------------------------- scorers and protocol
def fit_predict(arm, Xtr, ytr, Rtr, Xte):
    cols = slice(0, 9) if arm in ('E042', 'E042R', 'LGB9R') else slice(0, 41)
    w = np.ones(len(ytr)) if arm in ('E042', 'LR41') else Rtr.astype(np.float64)
    if arm in ('E042', 'E042R', 'LR41'):  # LR41 = E049 (logistic on all 41 columns, weight 1)
        if arm == 'LR41':
            assert cols == slice(0, 41) and np.all(w == 1), 'E049 must be logistic, 41 columns, weight 1'
        Xa, Xb = Xtr[:, cols].astype(np.float32), Xte[:, cols].astype(np.float32)
        const = Xa.std(0) < 1e-6  # decisions.md: near-constant columns in the fitting part are set to 0 on both sides
        if const.any():
            Xa, Xb = Xa.copy(), Xb.copy(); Xa[:, const] = 0; Xb[:, const] = 0
        return s2.fit_predict('logreg', Xa, ytr, w, Xb)
    import lightgbm as lgb
    m = lgb.train(LGB, lgb.Dataset(Xtr[:, cols], label=np.clip(ytr, 0, 1), weight=w), num_boost_round=300)
    return m.predict(Xte[:, cols])


def predict_split(imgs, fit_sel, pred_sel, arm):
    X = np.concatenate([imgs[i]['X'] for i in fit_sel]); y = np.concatenate([imgs[i]['u'] for i in fit_sel])
    R = np.concatenate([np.full(len(imgs[i]['u']), imgs[i]['R']) for i in fit_sel])
    Xe = np.concatenate([imgs[i]['X'] for i in pred_sel])
    p = fit_predict(arm, X, y, R, Xe)
    out, o = {}, 0
    for i in pred_sel:
        n = len(imgs[i]['u']); out[i] = p[o:o + n]; o += n
    return out


def nested(imgs, groups, half, arm, seed):
    keep, taus = {}, []
    for s, a in ((0, 1), (1, 0)):
        S = [i for i in range(len(imgs)) if half[i] == s]; A = [i for i in range(len(imgs)) if half[i] == a]
        ug = sorted({groups[i] for i in S})
        perm = np.random.default_rng(seed).permutation(len(ug))
        part = {g: int(perm[k] % r42.KIN) for k, g in enumerate(ug)}
        cross = {}
        for q in range(r42.KIN):
            ins = [i for i in S if part[groups[i]] == q]; trn = [i for i in S if part[groups[i]] != q]
            cross.update(predict_split(imgs, trn, ins, arm))
        tau = r42.best_tau(imgs, S, cross, r42.TAU_C2)
        pa = predict_split(imgs, S, A, arm)
        for i in A:
            keep[i] = pa[i] >= tau
        taus.append(tau)
    return keep, taus


def parts(imgs, keep):
    """per image (dS_sel, dD_sel, gt)."""
    return np.array([[imgs[i]['dS'][keep[i]].sum(), imgs[i]['dD'][keep[i]].sum(), imgs[i]['gt']] for i in range(len(imgs))])


def pq(P):
    return P[:, 0].sum() / (0.5 * P[:, 2].sum() + P[:, 1].sum())


def lofo(folds, groups, arm):
    res = {}
    for f in (0, 1, 2):
        tr = [im for g in (0, 1, 2) if g != f for im in folds[g]]
        trg = [gr for g in (0, 1, 2) if g != f for gr in groups[g]]
        idx = list(range(len(tr)))
        ug = sorted(set(trg)); perm = np.random.default_rng(0).permutation(len(ug))
        part = {g: int(perm[k] % r42.KIN) for k, g in enumerate(ug)}
        cross = {}
        for q in range(r42.KIN):
            ins = [i for i in idx if part[trg[i]] == q]; trn = [i for i in idx if part[trg[i]] != q]
            cross.update(predict_split(tr, trn, ins, arm))
        tau = r42.best_tau(tr, idx, cross, r42.TAU_C2)
        te = folds[f]
        allimgs = tr + te
        pr = predict_split(allimgs, idx, list(range(len(tr), len(allimgs))), arm)
        keep = {i: pr[len(tr) + i] >= tau for i in range(len(te))}
        res[f] = {'tau': tau, 'parts': parts(te, keep)}
    return res


def main():
    global e6, f1, g1, r42, s2
    ap = argparse.ArgumentParser(); ap.add_argument('--pilot', action='store_true'); a = ap.parse_args()
    import e006_maskrcnn_refiner as e6
    import f1_fusion as f1
    import g1_group_eval as g1
    import e042_rescore as r42
    import s2_rescore as s2
    OUT.mkdir(parents=True, exist_ok=True)
    if a.pilot:
        imgs, tm = build_fold(0, pilot=20)
        ncand = [len(im['u']) for im in imgs]
        t = time.time()
        X = np.concatenate([im['X'] for im in imgs]); y = np.concatenate([im['u'] for im in imgs])
        for arm in ARMS:
            fit_predict(arm, X, y, np.ones(len(y)), X)
        tfit = (time.time() - t) / len(ARMS)
        try:
            import psutil
            mem = psutil.Process().memory_info().peak_wset / 2 ** 30
        except Exception:
            mem = None
        print(json.dumps({'images': 20, 'features_s': tm['features'], 'features_s_per_image': tm['features'] / 20,
                          'projected_features_h': tm['features'] / 20 * 707 / 3600, 'candidates_per_image_max': max(ncand),
                          'candidates_per_image_mean': float(np.mean(ncand)), 'fit_s_on_pilot': tfit,
                          'peak_ws_gb': mem}, indent=1))
        return
    folds, groups, halves = {}, {}, {}
    for f in (0, 1, 2):
        folds[f] = build_fold(f)
        _, _, gr, hv, _ = g1.fold_data(f, workers=2)
        groups[f], halves[f] = list(gr), hv
    print(f'[{time.time() - T0:.0f}s] features ready', flush=True)
    P, taus = {}, {}
    for arm in ARMS:
        for s in SEEDS:
            for f in (0, 1, 2):
                keep, tt = nested(folds[f], groups[f], halves[f], arm, s)
                P[(arm, s, f)] = parts(folds[f], keep); taus[(arm, s, f)] = tt
                check_cap(f'nested {arm} seed {s} fold {f}')
            print(f'[{time.time() - T0:.0f}s] {arm} seed {s}: ' + ' '.join(
                f'{pq(P[(arm, s, f)]):.4f}' for f in (0, 1, 2)) + f' pooled {pq(np.concatenate([P[(arm, s, f)] for f in (0, 1, 2)])):.4f}', flush=True)
    pooled = {(arm, s): pq(np.concatenate([P[(arm, s, f)] for f in (0, 1, 2)])) for arm in ARMS for s in SEEDS}
    # E042 seed 0 must reproduce the cached E042 (official counts 0.4512256; additive identity)
    e042_ref = json.loads((W / 'preds/e042/result.json').read_text(encoding='utf-8'))['C2_vs_A5']['pq_b']
    print('E042 seed 0 reproduction', round(pooled[('E042', 0)], 7), 'vs', round(e042_ref, 7), flush=True)
    assert abs(pooled[('E042', 0)] - e042_ref) < 2e-4, 'E042 reproduction failed'
    # bootstrap of the seed-mean delta (shared group weights per fold across seeds and arms)
    rng = np.random.default_rng(0)
    gidx = {}
    for f in (0, 1, 2):
        ug = sorted(set(groups[f])); gi = {g: k for k, g in enumerate(ug)}
        gidx[f] = (np.array([gi[g] for g in groups[f]]), len(ug))
    res = {}
    for arm in ('E042R', 'LGB9R', 'E047'):
        dbar = float(np.mean([pooled[(arm, s)] - pooled[('E042', s)] for s in SEEDS]))
        perfold = {f: float(np.mean([pq(P[(arm, s, f)]) - pq(P[('E042', s, f)]) for s in SEEDS])) for f in (0, 1, 2)}
        reps = np.zeros(1000)
        rng = np.random.default_rng(0)
        for r in range(1000):
            wts = {f: rng.multinomial(gidx[f][1], np.ones(gidx[f][1]) / gidx[f][1]).astype(float)[gidx[f][0]] for f in (0, 1, 2)}
            ds = []
            for s in SEEDS:
                v = {}
                for b in (arm, 'E042'):
                    tot = sum((wts[f][:, None] * P[(b, s, f)]).sum(0) for f in (0, 1, 2))
                    v[b] = tot[0] / (0.5 * tot[2] + tot[1])
                ds.append(v[arm] - v['E042'])
            reps[r] = np.mean(ds)
        res[arm] = {'delta_seed_mean': dbar, 'per_fold_seed_mean': perfold, 'p_delta_gt0': float((reps > 0).mean()),
                    'ci90': [float(np.percentile(reps, 5)), float(np.percentile(reps, 95))],
                    'per_seed_pooled': {s: pooled[(arm, s)] for s in SEEDS},
                    'folds_positive': int(sum(v > 0 for v in perfold.values()))}
        print(f"{arm} vs E042: d {dbar:+.4f} P {res[arm]['p_delta_gt0']:.2f} ci90 {[round(x, 4) for x in res[arm]['ci90']]} "
              f"folds {[round(v, 4) for v in perfold.values()]}", flush=True)
    res['E042_per_seed_pooled'] = {s: pooled[('E042', s)] for s in SEEDS}
    res['taus'] = {f'{k[0]}|{k[1]}|{k[2]}': v for k, v in taus.items()}
    g = res['E047']
    gate_a = g['delta_seed_mean'] >= 0.0015 and g['p_delta_gt0'] >= 0.9 and g['folds_positive'] >= 2
    # LOFO transfer diagnostic (seed 0)
    lo = {arm: lofo(folds, groups, arm) for arm in ('E042', 'E047')}
    lpf = {f: pq(lo['E047'][f]['parts']) - pq(lo['E042'][f]['parts']) for f in (0, 1, 2)}
    lpool = pq(np.concatenate([lo['E047'][f]['parts'] for f in (0, 1, 2)])) - pq(np.concatenate([lo['E042'][f]['parts'] for f in (0, 1, 2)]))
    gate_b = lpool > 0 and min(lpf.values()) >= -0.004
    res['lofo'] = {'per_fold_delta': lpf, 'pooled_delta': lpool, 'taus': {arm: [lo[arm][f]['tau'] for f in (0, 1, 2)] for arm in lo},
                   'pq_e047': [pq(lo['E047'][f]['parts']) for f in (0, 1, 2)], 'pq_e042': [pq(lo['E042'][f]['parts']) for f in (0, 1, 2)]}
    print(f"LOFO E047 vs E042: pooled {lpool:+.4f} folds {[round(v, 4) for v in lpf.values()]}", flush=True)
    # feature importance (gain) on all OOF, seed 0 settings
    import lightgbm as lgb
    allim = [im for f in (0, 1, 2) for im in folds[f]]
    X = np.concatenate([im['X'] for im in allim]); y = np.concatenate([im['u'] for im in allim])
    R = np.concatenate([np.full(len(im['u']), im['R']) for im in allim])
    m = lgb.train(LGB, lgb.Dataset(X, label=np.clip(y, 0, 1), weight=R), num_boost_round=300)
    imp = sorted(zip(NAMES, m.feature_importance('gain')), key=lambda x: -x[1])
    res['importance_gain'] = [(n, float(v)) for n, v in imp]
    res['gate_a'], res['gate_b'] = bool(gate_a), bool(gate_b)
    res['adopt'] = bool(gate_a and gate_b)
    res['note'] = 'nested group halves; bootstrap conditional on fitted selections; LOFO = development diagnostic (seed 0)'
    (OUT / 'result.json').write_text(json.dumps(res, indent=1, default=str), encoding='utf-8')
    print('top features', [n for n, _ in imp[:10]])
    print('gate a', gate_a, 'gate b', gate_b, 'ADOPT', res['adopt'], f'[{time.time() - T0:.0f}s]')


if __name__ == '__main__':
    main()
