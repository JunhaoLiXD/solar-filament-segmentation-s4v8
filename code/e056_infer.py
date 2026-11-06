"""E056 (D1) 8-view OOF inference -> member pools in the f1 format (one member per dihedral view).
Instance rule (pre-registered): bilinear upsample to 2048, p > 0.5, 8-connected components, area >= 64 px at 2048,
score = mean p inside the component. Pools: preds/f1_fusion/pools/e056{view}_fold{f}.pkl (view '' = identity).
Checks: dihedral forward/inverse round-trip; identity view == the training run's saved single-view OOF probs (fresh reload).
--xplat-check (E064 Kaggle-trained runs only; Codex review docs/reviews/review-2026-10-31-e064-reload-check-codex.md): the
same-machine max |dp| < 0.02 assert is replaced by a per-image cross-platform check on the first 5 images (shape, finite,
range, max |dp| < 0.05, p99.99 |dp| < 0.005, binarised p > 0.5 IoU >= 0.995, both empty = 1); identity-view diffs of every
image are recorded in infer_receipt.json (anomaly = max |dp| >= 0.05 or IoU < 0.99, investigated before any score is seen).
Default path unchanged.
Run: ../../.venv-gpu/Scripts/python.exe code/e056_infer.py --fold 0"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
VIEWS = {'': ((), False), 'hf': ((1,), False), 'vf': ((0,), False), 'hvf': ((0, 1), False),
         'tr': ((), True), 'trhf': ((1,), True), 'trvf': ((0,), True), 'trhvf': ((0, 1), True)}  # flips first, then transpose
THR, MIN_AREA = 0.5, 64


def fwd(a, v):
    flips, tr = VIEWS[v]
    a = np.flip(a, axis=flips) if flips else a
    return a.T if tr else a


def inv(a, v):
    flips, tr = VIEWS[v]
    a = a.T if tr else a
    return np.flip(a, axis=flips) if flips else a


def load_input(path, size):
    from PIL import Image
    with Image.open(path) as im:
        g = im.convert('L')
        small = g.reduce(2048 // size) if 2048 % size == 0 else g.resize((size, size), Image.Resampling.BOX)
        return np.asarray(small, dtype=np.uint8).copy()


def instances(prob2048, e6):
    import cv2
    n, lab, stats, _ = cv2.connectedComponentsWithStats((prob2048 > THR).astype(np.uint8), connectivity=8)
    out = []
    for k in range(1, n):
        if stats[k, cv2.CC_STAT_AREA] < MIN_AREA:
            continue
        x0, y0, w, h = (int(stats[k, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        m = lab[y0:y0 + h, x0:x0 + w] == k
        out.append((e6.trim(y0, x0, m), float(prob2048[y0:y0 + h, x0:x0 + w][m].mean())))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, required=True)
    ap.add_argument('--run', default=None, help='training output dir (default runs/e056-convnext-fold{f})')
    ap.add_argument('--prefix', default='e056', help='pool name prefix (E059 D2: e059)')
    ap.add_argument('--xplat-check', action='store_true', help='cross-platform reload check (E064 Kaggle-trained runs)')
    args = ap.parse_args()
    import torch, torch.nn.functional as F
    import segmentation_models_pytorch as smp
    import e006_maskrcnn_refiner as e6, g1_group_eval as g1
    f = args.fold; run = W / (args.run or f'runs/e056-convnext-fold{f}')
    for v in VIEWS:  # dihedral round trip
        a = np.arange(12).reshape(3, 4)
        assert np.array_equal(inv(fwd(a, v), v), a), v
    ck = torch.load(run / f'e004_fold{f}_unet.pt', map_location='cpu', weights_only=False)
    cfg = ck['config']; size = int(cfg['size'])
    model = smp.Unet(encoder_name=cfg['encoder'], encoder_weights=None, in_channels=1, classes=1)
    model.load_state_dict(ck['state_dict'], strict=True)
    dev = torch.device('cuda'); torch.backends.cudnn.benchmark = False
    model = model.to(dev).to(memory_format=torch.channels_last).eval()
    ids, _, _, _, paths = g1.fold_data(f, workers=2)
    tr_ids = json.loads((run / f'e004_fold{f}_oof_ids.json').read_text(encoding='utf-8'))
    assert sorted(tr_ids) == sorted(ids), 'OOF id sets differ'
    ref = np.load(run / f'e004_fold{f}_oof_probs.npy', mmap_mode='r'); rrow = {p: k for k, p in enumerate(tr_ids)}
    xp = []  # --xplat-check: per-image identity-view diagnostics
    if args.xplat_check:
        assert len(set(tr_ids)) == len(tr_ids) and len(set(ids)) == len(ids), 'duplicate ids'
        assert ref.shape == (len(tr_ids), size, size), f'reference OOF shape {ref.shape}'
    pools = {v: [] for v in VIEWS}; t0 = time.time(); maxdiff = 0.0
    with torch.inference_mode():
        for i, pid in enumerate(ids):
            img = load_input(paths[pid], size)
            for v in VIEWS:
                x = torch.from_numpy(np.ascontiguousarray(fwd(img, v))).to(dev).float()[None, None] / 255
                x = ((x - x.mean()) / (x.std() + 1e-6)).contiguous(memory_format=torch.channels_last)
                with torch.autocast('cuda', dtype=torch.float16):
                    p = model(x).float().sigmoid()[0, 0].cpu().numpy().astype(np.float16)
                p = np.ascontiguousarray(inv(p, v))
                if v == '' and args.xplat_check:
                    r = np.asarray(ref[rrow[pid]], dtype=np.float32); q = p.astype(np.float32)
                    assert r.shape == q.shape and np.isfinite(r).all() and np.isfinite(q).all(), f'shape / finite {pid}'
                    assert r.min() >= 0 and r.max() <= 1 and q.min() >= 0 and q.max() <= 1, f'range {pid}'
                    d = np.abs(q - r); bq, br = q > THR, r > THR; un = int((bq | br).sum())
                    xp.append({'id': pid, 'max': float(d.max()), 'p9999': float(np.percentile(d, 99.99)),
                               'iou': float((bq & br).sum() / un) if un else 1.0, 'flips': int((bq != br).sum())})
                    if i < 5:
                        e = xp[-1]
                        assert e['max'] < 0.05 and e['p9999'] < 0.005 and e['iou'] >= 0.995, f'xplat reload check failed {e}'
                elif v == '' and i < 5:
                    maxdiff = max(maxdiff, float(np.abs(p.astype(np.float32) - ref[rrow[pid]].astype(np.float32)).max()))
                up = F.interpolate(torch.from_numpy(p.astype(np.float32))[None, None].to(dev), size=(2048, 2048),
                                   mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
                pools[v].append(instances(up, e6))
            if i == 4 and args.xplat_check:
                print(f'xplat reload check ok {xp[:5]}', flush=True)
            elif i == 4:
                assert maxdiff < 0.02, f'identity view differs from training OOF: {maxdiff}'
                print(f'reload check ok (max |dp| {maxdiff:.4f})', flush=True)
            if i % 50 == 0:
                print(f'fold {f} image {i}/{len(ids)} [{time.time() - t0:.0f}s]', flush=True)
    out = W / 'preds/f1_fusion/pools'; out.mkdir(parents=True, exist_ok=True)
    for v in VIEWS:
        (out / f'{args.prefix}{v}_fold{f}.pkl').write_bytes(pickle.dumps(pools[v], protocol=4))
    stats = {v: float(np.mean([len(x) for x in pools[v]])) for v in VIEWS}
    rec = {'fold': f, 'images': len(ids), 'seconds': time.time() - t0,
           'reload_maxdiff': maxdiff, 'inst_per_image': stats,
           'rule': {'thr': THR, 'min_area': MIN_AREA, 'conn': 8}}
    if args.xplat_check:
        import hashlib, platform, timm
        sha = lambda q: hashlib.sha256(Path(q).read_bytes()).hexdigest()
        anom = [e for e in xp if e['max'] >= 0.05 or e['iou'] < 0.99]
        rec['xplat'] = {'mode': 'xplat', 'thresholds': {'first_n': 5, 'max': 0.05, 'p9999': 0.005, 'iou': 0.995},
                        'anomaly_rule': {'max': 0.05, 'iou': 0.99}, 'n_anomalies': len(anom), 'anomalies': anom,
                        'all_images': xp, 'worst_iou': min(e['iou'] for e in xp), 'worst_max': max(e['max'] for e in xp),
                        'checkpoint_sha256': sha(run / f'e004_fold{f}_unet.pt'),
                        'ref_oof_sha256': sha(run / f'e004_fold{f}_oof_probs.npy'), 'infer_source_sha256': sha(__file__),
                        'gpu': torch.cuda.get_device_name(0), 'versions': {'torch': torch.__version__, 'timm': timm.__version__,
                        'smp': smp.__version__, 'cuda': torch.version.cuda, 'cudnn': torch.backends.cudnn.version(),
                        'numpy': np.__version__, 'python': platform.python_version()}}
        print(f"xplat: {len(anom)} anomalous images; worst IoU {rec['xplat']['worst_iou']:.4f}, worst max {rec['xplat']['worst_max']:.4f}", flush=True)
    (run / 'infer_receipt.json').write_text(json.dumps(rec, indent=1), encoding='utf-8')
    print('done', stats, f'[{time.time() - t0:.0f}s]')


if __name__ == '__main__':
    main()
