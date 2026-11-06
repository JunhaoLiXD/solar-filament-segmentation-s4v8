"""Agreement step 2: rescoring pilot on the candidate tables (code/s1_candidates.py).

Target: u = dS / records (per-record metric numerator contribution). Keep a candidate when predicted u >= tau.
Leakage control (Codex review): everything happens INSIDE each original held-out fold (one fixed base model).
Groups of the fold are split into K parts; for each part k: fit on the other parts, choose tau on cross-fitted
predictions of those parts (inner group CV), assess on part k. Every group is assessed exactly once.
Scores: summed dS / (0.5 GT + sum dD) over assessed images. Comparators: fixed v2 (score >= 0.8) and
threshold-only (score threshold chosen the same way).
Models (torch, no extra packages): 'logreg' (primary, pre-declared), 'mlp' (exploratory).

Run: ../../.venv-gpu/Scripts/python.exe code/s2_rescore.py [--tag ""] [--seeds 0 1 2]
"""
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

WORKDIR = Path(__file__).resolve().parent.parent
K = 5
TAUS = np.round(np.arange(0.02, 0.80, 0.01), 3)
SCORE_TAUS = np.round(np.arange(0.40, 0.995, 0.01), 3)


def load(fold, tag):
    d = WORKDIR / f'preds/s1_candidates/fold{fold}{tag}'
    c = list(csv.DictReader(open(d / 'candidates.csv', encoding='utf-8')))
    im = {r['image']: r for r in csv.DictReader(open(d / 'images.csv', encoding='utf-8'))}
    return c, im


def features(rows):
    by_img = defaultdict(list)
    for r in rows:
        by_img[r['image']].append(float(r['score']))
    X = []
    for r in rows:
        s = min(max(float(r['score']), 1e-4), 1 - 1e-4)
        sc = by_img[r['image']]
        a = float(r['area'])
        X.append([math.log(s / (1 - s)), s, float(r['det_mean_level']), float(r['refiner_fallback']),
                  float(r['refiner_max']), float(r['refiner_mean']), math.log(a), math.log(float(r['h'])),
                  math.log(float(r['w'])), math.log(max(float(r['thickness']), 1e-3)), float(r['fill']),
                  math.log(max(float(r['h']), float(r['w'])) / min(float(r['h']), float(r['w']))),
                  float(r['r_over_R']), float(r['r_over_R']) ** 2,
                  math.log1p(sum(x >= 0.8 for x in sc)), math.log1p(len(sc)),
                  sum(x > float(r['score']) for x in sc) / len(sc)])
    return np.asarray(X, np.float32)


FEATURE_NAMES = ['logit_score', 'score', 'det_mean_level', 'refiner_fallback', 'refiner_max', 'refiner_mean',
                 'log_area', 'log_h', 'log_w', 'log_thickness', 'fill', 'log_aspect', 'r_over_R', 'r_over_R2',
                 'log1p_n_hi_in_image', 'log1p_n_cand_in_image', 'score_rank_frac']


def fit_predict(kind, Xtr, ytr, wtr, Xte, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    xt = torch.tensor((Xtr - mu) / sd); xe = torch.tensor((Xte - mu) / sd)
    y = torch.tensor(np.clip(ytr, 0, 1), dtype=torch.float32); w = torch.tensor(wtr, dtype=torch.float32)
    if kind == 'logreg':
        model = torch.nn.Linear(xt.shape[1], 1)
        opt = torch.optim.LBFGS(model.parameters(), max_iter=200, line_search_fn='strong_wolfe')
        wd = 1e-3

        def closure():
            opt.zero_grad()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(xt)[:, 0], y, reduction='none') * w).sum() / w.sum()
            loss = loss + wd * (model.weight ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)
    else:
        model = torch.nn.Sequential(torch.nn.Linear(xt.shape[1], 32), torch.nn.ReLU(), torch.nn.Dropout(0.1),
                                    torch.nn.Linear(32, 1))
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-2)
        for _ in range(400):
            model.train(); opt.zero_grad()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(xt)[:, 0], y, reduction='none') * w).sum() / w.sum()
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(xe)[:, 0]).numpy()


def pq_sel(dS, dD, gt, keep):
    return dS[keep].sum() / (0.5 * gt + dD[keep].sum())


def best_tau(pred, dS, dD, gt, taus):
    vals = [pq_sel(dS, dD, gt, pred >= t) for t in taus]
    return taus[int(np.argmax(vals))]


def run_fold(rows, imgs, pool, seed):
    if pool == 'v2':
        rows = [r for r in rows if float(r['score']) >= 0.8]
    X = features(rows)
    dS = np.array([float(r['dS']) for r in rows]); dD = np.array([float(r['dD']) for r in rows])
    rec = np.array([float(r['records']) for r in rows]); u = dS / rec
    score = np.array([float(r['score']) for r in rows])
    img = np.array([r['image'] for r in rows])
    groups = sorted({imgs[i]['group'] for i in imgs})
    rng = np.random.default_rng(seed)
    part_of = {g: k % K for k, g in enumerate(rng.permutation(groups))}
    img_part = {i: part_of[imgs[i]['group']] for i in imgs}
    cpart = np.array([img_part[i] for i in img])
    gt_part = np.zeros(K)
    for i, r in imgs.items():
        gt_part[img_part[i]] += float(r['n_gt'])
    keep = {m: np.zeros(len(rows), bool) for m in ('v2', 'thr', 'logreg', 'mlp')}
    keep['v2'] = score >= 0.8
    taus_used = defaultdict(list)
    for k in range(K):
        tr, te = cpart != k, cpart == k
        gt_tr = gt_part[np.arange(K) != k].sum()
        # threshold-only comparator
        t = best_tau(score[tr], dS[tr], dD[tr], gt_tr, SCORE_TAUS)
        keep['thr'][te] = score[te] >= t; taus_used['thr'].append(float(t))
        for m in ('logreg', 'mlp'):
            inner = np.zeros(tr.sum())
            tri = np.flatnonzero(tr); pin = cpart[tri]
            for j in set(pin.tolist()):
                a, b = tri[pin != j], tri[pin == j]
                inner[pin == j] = fit_predict(m, X[a], u[a], rec[a], X[b], seed)
            t = best_tau(inner, dS[tri], dD[tri], gt_tr, TAUS)
            keep[m][te] = fit_predict(m, X[tri], u[tri], rec[tri], X[te], seed) >= t
            taus_used[m].append(float(t))
    gt = gt_part.sum()
    return {m: (dS[v].sum(), dD[v].sum(), gt) for m, v in keep.items()}, taus_used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', default='')
    ap.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    a = ap.parse_args()
    data = {f: load(f, a.tag) for f in (0, 1, 2)}
    report = {}
    for pool in ('v2', 'pool0.4'):
        per_seed = []
        for seed in a.seeds:
            res = {f: run_fold(*data[f], pool, seed) for f in (0, 1, 2)}
            row = {}
            for m in ('v2', 'thr', 'logreg', 'mlp'):
                S = sum(res[f][0][m][0] for f in res); D = sum(res[f][0][m][1] + 0.5 * res[f][0][m][2] for f in res)
                row[m] = {'3fold': S / D, 'folds': [res[f][0][m][0] / (res[f][0][m][1] + 0.5 * res[f][0][m][2]) for f in (0, 1, 2)]}
            row['taus'] = {m: [res[f][1][m] for f in (0, 1, 2)] for m in ('thr', 'logreg', 'mlp')}
            per_seed.append(row)
            print(pool, 'seed', seed, {m: round(row[m]['3fold'], 4) for m in ('v2', 'thr', 'logreg', 'mlp')},
                  {m: [round(x, 4) for x in row[m]['folds']] for m in ('logreg', 'mlp')}, flush=True)
        report[pool] = {'per_seed': per_seed,
                        'mean_3fold': {m: float(np.mean([r[m]['3fold'] for r in per_seed])) for m in ('v2', 'thr', 'logreg', 'mlp')}}
    out = WORKDIR / f'preds/s2_rescore{a.tag}.json'
    out.write_text(json.dumps(report, indent=1), encoding='utf-8')
    print(json.dumps({p: report[p]['mean_3fold'] for p in report}, indent=1))


if __name__ == '__main__':
    main()
