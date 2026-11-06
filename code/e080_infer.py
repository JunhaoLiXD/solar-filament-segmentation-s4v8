"""S4V8 deployment entry (E080): 18 checkpoints + saved C2 and tau -> submission CSV. No training labels, OOF data or cached predictions are read.
Members follow params.json: family -> fold -> view. Each list is fused (MATCH 0.5, floor 0.30), scored by the saved logistic C2, kept at the saved tau.
Run: python code/e080_infer.py --images <dir with test jpg> --out submission.csv [--limit N] [--device cuda] [--params params.json]
The bundle root is the parent of this file's folder; weights in weights/, parameters in params.json. --params selects another parameter
file (for example params_retrained.json from train/collect.py); its "weights_sha256" names the hash list checked before loading."""
import argparse, csv, hashlib, json, os, pickle, sys, time, zlib
from pathlib import Path
import numpy as np
from pycocotools import mask as cm
HERE = Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
os.environ.setdefault('HF_HUB_OFFLINE', '1')


def features(lists, fam, view, nfam, nview, gray, e6, m47, r42):
    """same as code/e078_test.py features: the 9 C2 columns of each fused instance."""
    M = len(lists)
    cls = m47.fuse_any(lists, M)
    masks, idx = r42.finalize_idx(cls, r42.FLOOR)
    X = []
    for c, j in zip(masks, idx):
        fs, _, mem = cls[j]; ks = list(mem); sc = [s for _, s in mem.values()]
        yy, xx = np.nonzero(c[2]); cy, cx = c[0] + yy.mean(), c[1] + xx.mean()
        X.append([fs, len(ks) / M, float(np.mean(sc)), len({fam[k] for k in ks}) / nfam, len({view[k] for k in ks}) / nview,
                  np.log1p(r42.area(c)), r42.contrast(gray, c), float(np.hypot(cy - 1024, cx - 1024) / 1024), np.log1p(len(masks))])
    return masks, np.array(X, np.float64).reshape(-1, 9)


def ram_peak_gib():
    """peak working set of this process, kept by the OS (Windows and Linux, no extra package)."""
    if os.name != 'nt':
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / 2 ** 30
    import ctypes
    from ctypes import wintypes

    class C(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
            'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage', 'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
            'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    c = C(); c.cb = ctypes.sizeof(C)
    k, ps = ctypes.WinDLL('kernel32'), ctypes.WinDLL('psapi')
    k.GetCurrentProcess.restype = wintypes.HANDLE
    ps.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(C), wintypes.DWORD]
    assert ps.GetProcessMemoryInfo(k.GetCurrentProcess(), ctypes.byref(c), c.cb)
    return c.PeakWorkingSetSize / 2 ** 30


def list_images(folder, limit=0):
    """sorted test images by stem; a repeated stem is an error."""
    found = [p for p in sorted(Path(folder).rglob('*')) if p.suffix.lower() in ('.jpg', '.jpeg')]
    paths = {p.stem: p for p in found}
    assert len(paths) == len(found), 'repeated image stem'
    ids = sorted(paths)[:limit or None]
    assert ids, f'no images under {folder}'
    return ids, paths


def check_weights(sha_list='weights.sha256'):
    """every checkpoint must match the hash list (default weights.sha256) before anything is deserialized."""
    for line in (ROOT / sha_list).read_text(encoding='utf-8').splitlines():
        want, name = line.split('  ')
        h = hashlib.sha256()
        with open(ROOT / name, 'rb') as fh:
            for b in iter(lambda: fh.read(1 << 20), b''):
                h.update(b)
        assert h.hexdigest() == want, f'hash mismatch {name}'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--images', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0); ap.add_argument('--device', default='cuda'); ap.add_argument('--params', default='params.json')
    ap.add_argument('--dump', default='', help='also write each member list here, for an external comparison; never read back'); args = ap.parse_args()
    import torch, s1_candidates as s1, t1_tta
    import segmentation_models_pytorch as smp
    import e006_maskrcnn_refiner as e6, e042_rescore as r42, e047_c2plus as m47, f1_fusion as f1
    import validate_submission as vs
    from e056_test_infer import predict
    m47.f1, m47.r42, m47.e6 = f1, r42, e6
    torch.backends.cudnn.benchmark = False
    torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])  # recorded in the dense checkpoints
    dev = torch.device(args.device)
    t0 = time.time(); stats = {}
    par = json.loads((ROOT / args.params).read_text(encoding='utf-8'))
    check_weights(par.get('weights_sha256', 'weights.sha256'))
    ids, paths = list_images(args.images, args.limit)
    views = par['views']
    pools = {}
    for mm in par['models']:
        t1 = time.time()
        if mm['kind'] == 'dense':
            ck = torch.load(ROOT / mm['weights'], map_location='cpu', weights_only=True); cfg = ck['config']
            model = smp.Unet(encoder_name=cfg['encoder'], encoder_weights=None, in_channels=1, classes=1)
            model.load_state_dict(ck['state_dict'], strict=True)
            model = model.to(dev).to(memory_format=torch.channels_last).eval()
            with torch.inference_mode():
                prs = [predict(model, paths[p], int(cfg['size']), dev, e6) for p in ids]
            for v in views:
                pools[(mm['id'], v)] = [x[v] for x in prs]
            del model, prs
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
                pools[(mm['id'], v)] = mp
            del det, ref
        torch.cuda.empty_cache()
        if args.dump:
            Path(args.dump).mkdir(parents=True, exist_ok=True)
            for v in views:
                (Path(args.dump) / f"{mm['id']}__{v or 'id'}.pkl.z").write_bytes(zlib.compress(pickle.dumps(pools[(mm['id'], v)], protocol=4), 6))
        print(f"[{time.time() - t0:.0f}s] {mm['id']} {time.time() - t1:.0f}s", flush=True)
    # fusion order: family -> fold -> view, as in params.json
    order = [(mm['id'], v) for mm in par['models'] for v in views]
    fam = [mm['family'] for mm in par['models'] for _ in views]
    view = [k for _ in par['models'] for k in range(len(views))]
    assert len(order) == par['members'], (len(order), par['members'])
    c2, tau = par['c2'], par['tau']
    mu, sd, wt = (np.array(c2[k], np.float32) for k in ('mu', 'sd', 'weight'))
    rows, t2 = [], time.time()
    for i, pid in enumerate(ids):
        masks, X = features([pools[k][i] for k in order], fam, view, par['families'], len(views), e6.load_gray(paths[pid]), e6, m47, r42)
        p = 1 / (1 + np.exp(-(((X.astype(np.float32) - mu) / sd) @ wt + c2['bias']))) if len(masks) else np.zeros(0)
        k = 0
        for c, q in zip(masks, p):
            if q >= tau:
                k += 1
                rows.append((f'{pid}_{k}', cm.encode(np.asfortranarray(e6.dense(c)))['counts'].decode('ascii')))
    with open(args.out, 'w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh); w.writerow(['filament_id', 'segmentation_rle']); w.writerows(rows)
    stats.update(valid=vs.validate_submission(str(args.out), allowed_ids=ids, reject_empty_masks=True, reject_all_empty=True)['valid'], images=len(ids), rows=len(rows), seconds=time.time() - t0, fusion_seconds=time.time() - t2, seconds_per_image=(time.time() - t0) / len(ids),
                 sha256=e6.sha_file(args.out), ram_peak_gib=ram_peak_gib(), vram_peak_gib=torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else None)
    print(json.dumps(stats), flush=True)
    Path(args.out).with_suffix('.stats.json').write_text(json.dumps(stats, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
