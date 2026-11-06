"""Step 3 of training: out-of-fold candidates of the 12 fold models, then the C2 rescorer and its threshold tau.

For each fold f, the four fold-f models (ep24, e034, D1, D2), never trained on fold f, run on the fold-f training images in the
same 8 views as at inference (32 members per image). The members are fused exactly as code/e080_infer.py does (IoU 0.5, floor 0.30)
and each fused instance gets the same 9 features. Its target is its IoU-weighted match to the annotator records of that image.
C2 = logistic regression on the 9 standardised features, fitted on all out-of-fold candidates of the three folds. tau is chosen by
cross-fitting over 5 parts of the image groups in folds.csv (grid 0.02..0.80). Writes the same "c2" and "tau" fields as params.json.

--evaluate also reports leave-one-fold-out PQ of the C2 layer: C2 and tau fitted on the candidates of two folds, applied to the third
(local PQ semantics of code/pq_metric.py over every annotator record). The base models behind the two fitting folds were trained with
the evaluated fold included, so this is not a fully nested cross-validation; 32 members per image, while deployment uses 96.
It is our own estimate on development data, not the leaderboard score.

Run from the repository root (weights assembled first):
  python train/fit_c2.py --data <folder with train_images and the train annotation JSON> --out params_refit.json [--evaluate]
  --limit N: N images per fold (smoke test only; the fitted numbers are then meaningless).
  --params: parameter file whose weights are used (default params.json; params_retrained.json after train/collect.py)."""
import argparse, csv, json, pickle, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / 'code'))
FAMILIES = ('ep24', 'e034', 'D1', 'D2')  # fusion order within a fold; D1 and D2 count as one family (3 families)


def fold_images(data, folds_csv, limit):
    """-> {fold: (ids, paths, entries, groups)}; entries = GT crops per annotator record, as the PQ code expects."""
    import e006_maskrcnn_refiner as e6
    ann_path, images = e6.discover(Path(data))
    paths = {s: p for s, (split, p) in images.items() if split == 'train'}
    rec_phys, rec_anns = e6.load_annotations(ann_path, images)
    recs = {}
    for rid, pid in sorted(rec_phys.items()):
        if rid in rec_anns:
            recs.setdefault(pid, []).append(rid)
    fold_of, group_of = {}, {}
    with open(folds_csv, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            fold_of[r['physical_id']] = int(r['fold']); group_of[r['physical_id']] = r['group_id']
    out = {}
    for f in (0, 1, 2):
        ids = sorted(p for p in fold_of if fold_of[p] == f)[:limit or None]
        assert all(p in paths for p in ids), 'a folds.csv image is missing from --data'
        tasks = [(paths[p], None, [[a['segmentation'] for a in rec_anns[r]] for r in recs[p]]) for p in ids]
        entries = [e6.prep_task(t)[3] for t in tasks]
        out[f] = (ids, paths, entries, [group_of[p] for p in ids])
    return out


def member_lists(mm, views, ids, paths, dev, e6, s1, t1_tta, smp, torch, predict):
    """one model on its images in every view -> {view: [per image [(crop, score)]]}; same code path as code/e080_infer.py."""
    out = {}
    if mm['kind'] == 'dense':
        ck = torch.load(ROOT / mm['weights'], map_location='cpu', weights_only=True); cfg = ck['config']
        model = smp.Unet(encoder_name=cfg['encoder'], encoder_weights=None, in_channels=1, classes=1)
        model.load_state_dict(ck['state_dict'], strict=True)
        model = model.to(dev).to(memory_format=torch.channels_last).eval()
        with torch.inference_mode():
            prs = [predict(model, paths[p], int(cfg['size']), dev, e6) for p in ids]
        for v in views:
            out[v] = [x[v] for x in prs]
        del model
    else:
        size = int(mm['size'])
        e6.INPUT_MODE = mm['input_mode']
        det, _ = e6.load_detector(ROOT / mm['weights'], dev, size); ref, _ = e6.load_refiner(ROOT / mm['refiner'], dev)
        factor = e6.Fraction(e6.FULL, size)
        for v in views:
            raw = e6.detector_raw(det, ids, paths, dev, size) if v == '' else t1_tta.view_raw(det, ids, paths, dev, size, v)
            per = e6.raw_by_image(raw, len(ids)) if v == '' else raw
            mp = []
            for i, pid in enumerate(ids):
                cache = e6.RefinerCache(ref, dev, e6.load_gray(paths[pid]))
                mp.append([(c, pl['score']) for c, pl in s1.pool(per[i], cache, 0.4, factor)])
            out[v] = mp
        del det, ref
    torch.cuda.empty_cache()
    return out


def candidates(lists, fam, view, nfam, nview, gray, ent, e6, m47, r42):
    """fused instances of one image with the 9 C2 features (as code/e080_infer.py) and their match to the annotator records."""
    M = len(lists)
    cls = m47.fuse_any(lists, M)
    masks, idx = r42.finalize_idx(cls, r42.FLOOR)
    R = len(ent)
    dS = np.zeros(len(masks)); dD = np.zeros(len(masks))
    for g in ent:
        iou = e6.overlap(g, masks) if len(masks) else np.zeros((len(g), 0))
        hit = iou > 0.5; k = hit.sum(0) if len(g) else np.zeros(len(masks), int)
        dS += np.where(hit, iou, 0).sum(0) if len(g) else 0
        dD += np.where(k > 0, 0.5 * k, 0.5)
    X = []
    for c, j in zip(masks, idx):
        fs, _, mem = cls[j]; ks = list(mem); sc = [s for _, s in mem.values()]
        yy, xx = np.nonzero(c[2]); cy, cx = c[0] + yy.mean(), c[1] + xx.mean()
        X.append([fs, len(ks) / M, float(np.mean(sc)), len({fam[k] for k in ks}) / nfam, len({view[k] for k in ks}) / nview,
                  np.log1p(r42.area(c)), r42.contrast(gray, c), float(np.hypot(cy - 1024, cx - 1024) / 1024), np.log1p(len(masks))])
    return {'X': np.array(X, np.float64).reshape(-1, 9), 'dS': dS, 'dD': dD, 'u': np.clip(dS / max(R, 1), 0, 1),
            'gt': sum(len(g) for g in ent), 'R': R}


def fit_global(imgs, groups, r42):
    """tau cross-fitted over 5 group parts (seed 0), then C2 fitted on all candidates."""
    idx = list(range(len(imgs)))
    ug = sorted(set(groups)); perm = np.random.default_rng(0).permutation(len(ug))
    part = {g: int(perm[k] % r42.KIN) for k, g in enumerate(ug)}
    cross = {}
    for q in range(r42.KIN):
        ins = [i for i in idx if part[groups[i]] == q]; trn = [i for i in idx if part[groups[i]] != q]
        cross.update(r42.predict_split(imgs, trn, ins, 'u'))
    tau = r42.best_tau(imgs, idx, cross, r42.TAU_C2)
    X, y = r42.fit_on(imgs, idx, 'u')
    return X.astype(np.float32), y, tau


def fit_params(X, y, torch):
    """logistic regression (LBFGS, L2 1e-3, unit weights) with its parameters returned."""
    torch.manual_seed(0)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    xt = torch.tensor((X - mu) / sd); t = torch.tensor(np.clip(y, 0, 1), dtype=torch.float32); w = torch.ones(len(y), dtype=torch.float32)
    model = torch.nn.Linear(xt.shape[1], 1)
    opt = torch.optim.LBFGS(model.parameters(), max_iter=200, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        loss = (torch.nn.functional.binary_cross_entropy_with_logits(model(xt)[:, 0], t, reduction='none') * w).sum() / w.sum()
        loss = loss + 1e-3 * (model.weight ** 2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {'mu': mu.tolist(), 'sd': sd.tolist(), 'weight': model.weight.detach().numpy()[0].tolist(), 'bias': float(model.bias.detach().numpy()[0])}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--data', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--folds-csv', default=str(ROOT / 'train/folds.csv')); ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--evaluate', action='store_true'); ap.add_argument('--cache', default='', help='save / reuse the candidates here')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--params', default='params.json'); args = ap.parse_args()
    import torch, s1_candidates as s1, t1_tta
    import segmentation_models_pytorch as smp
    import e006_maskrcnn_refiner as e6, e042_rescore as r42, e047_c2plus as m47, f1_fusion as f1
    from e056_test_infer import predict
    from e080_infer import check_weights
    m47.f1, m47.r42, m47.e6 = f1, r42, e6
    torch.backends.cudnn.benchmark = False
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])
    dev = torch.device(args.device); t0 = time.time()
    par = json.loads((ROOT / args.params).read_text(encoding='utf-8'))
    views = par['views']
    check_weights(par.get('weights_sha256', 'weights.sha256'))
    data = fold_images(args.data, args.folds_csv, args.limit)
    cache = Path(args.cache) if args.cache else None
    import hashlib
    fp = {'params': hashlib.sha256((ROOT / args.params).read_bytes()).hexdigest(), 'folds_csv': hashlib.sha256(Path(args.folds_csv).read_bytes()).hexdigest(),
          'weights': hashlib.sha256((ROOT / par.get('weights_sha256', 'weights.sha256')).read_bytes()).hexdigest(),  # the checkpoints actually used
          'limit': args.limit, 'ids': {f: data[f][0] for f in (0, 1, 2)}}
    if cache and cache.exists():
        saved = pickle.loads(cache.read_bytes())
        assert saved.get('fingerprint') == fp, f'cache {cache} was made for other models, folds, images or limit; delete it'
        C = saved['C']
    else:
        C = {}
        for f in (0, 1, 2):
            ids, paths, entries, _ = data[f]
            models = {mm['id'].split('_')[0]: mm for mm in par['models'] if mm['fold'] == f}
            lists = {}
            for fam in FAMILIES:
                got = member_lists(models[fam], views, ids, paths, dev, e6, s1, t1_tta, smp, torch, predict)
                for v in views:
                    lists[(fam, v)] = got[v]
                print(f'[{time.time() - t0:.0f}s] fold {f} {fam}', flush=True)
            keys = [(fam, v) for fam in FAMILIES for v in views]
            fam_i = [min(FAMILIES.index(a), 2) for a, _ in keys]; view_i = [views.index(b) for _, b in keys]
            C[f] = [candidates([lists[k][i] for k in keys], fam_i, view_i, par['families'], len(views), e6.load_gray(paths[ids[i]]),
                               entries[i], e6, m47, r42) for i in range(len(ids))]
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True); cache.write_bytes(pickle.dumps({'fingerprint': fp, 'C': C}, protocol=4))
    imgs = [im for f in (0, 1, 2) for im in C[f]]
    groups = [g for f in (0, 1, 2) for g in data[f][3]]
    X, y, tau = fit_global(imgs, groups, r42)
    assert args.limit or X.std(0).min() >= 1e-6, f"constant feature column: {X.std(0).tolist()}"
    res = {'tau': tau, 'c2': fit_params(X, y, torch), 'oof_images': len(imgs), 'oof_candidates': int(len(X)), 'limit': args.limit}
    ref = par['c2']
    res['compared_with'] = args.params
    res['vs_params_json'] = {'tau_equal': tau == par['tau'],
                             'max_abs_diff': {k: float(np.abs(np.array(res['c2'][k]) - np.array(ref[k])).max()) for k in ('mu', 'sd', 'weight')}
                             | {'bias': abs(res['c2']['bias'] - ref['bias'])}}
    if args.evaluate:
        per = {}
        for f in (0, 1, 2):
            tr = [im for g in (0, 1, 2) if g != f for im in C[g]]
            trg = [gr for g in (0, 1, 2) if g != f for gr in data[g][3]]
            Xf, yf, tf = fit_global(tr, trg, r42)
            pf = fit_params(Xf, yf, torch)
            mu, sd, wt = (np.array(pf[k], np.float32) for k in ('mu', 'sd', 'weight'))
            keep = []
            for im in C[f]:
                p = 1 / (1 + np.exp(-(((im['X'].astype(np.float32) - mu) / sd) @ wt + pf['bias']))) if len(im['X']) else np.zeros(0)
                keep.append(p >= tf)
            P = m47.parts(C[f], keep)
            per[f] = {'tau': tf, 'pq': float(m47.pq(P)), 'dS': float(P[:, 0].sum()), 'dD': float(P[:, 1].sum()), 'gt': float(P[:, 2].sum())}
        pooled = sum(per[f]['dS'] for f in per) / (0.5 * sum(per[f]['gt'] for f in per) + sum(per[f]['dD'] for f in per))
        res['evaluation'] = {'leave_one_fold_out_pq': per, 'pooled_pq': float(pooled),
                             'note': ('development data, local PQ semantics; not the leaderboard score. Leave one fold out applies to the C2 layer only: '
                                      'C2 and tau fitted on two folds of fixed out-of-fold candidates, whose base models saw the evaluated fold; '
                                      '32 members per image (deployment uses 96).')}
    res['seconds'] = time.time() - t0
    Path(args.out).write_text(json.dumps(res, indent=1), encoding='utf-8')
    print(json.dumps({k: v for k, v in res.items() if k != 'c2'}, indent=1))


if __name__ == '__main__':
    main()
