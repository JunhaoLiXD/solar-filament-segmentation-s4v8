"""E056 (D1) test inference: 3 fold models x 8 dihedral views -> 24 test member pools (same rule as code/e056_infer.py).
Out: preds/f2_test/_pools/e056_fold{f}_{view}.pkl (view 'id' = identity), test ids sorted as e052_a6c2_test.py.
Check: the first 3 OOF images of each fold re-run through the same path reproduce the cached OOF pools
(preds/f1_fusion/pools/e056{view}_fold{f}.pkl): same instance count, mask IoU >= 0.99, |score diff| < 1e-3.
Run: ../../.venv-gpu/Scripts/python.exe code/e056_test_infer.py [--prefix e059 --runs runs/e059-convnext-s2-fold{f}]"""
import argparse, json, pickle, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; W = HERE.parent
sys.path.insert(0, str(HERE))
from e056_infer import VIEWS, fwd, inv, load_input, instances


def predict(model, path, size, dev, e6):
    import torch, torch.nn.functional as F
    img = load_input(path, size); out = {}
    for v in VIEWS:
        x = torch.from_numpy(np.ascontiguousarray(fwd(img, v))).to(dev).float()[None, None] / 255
        x = ((x - x.mean()) / (x.std() + 1e-6)).contiguous(memory_format=torch.channels_last)
        with torch.autocast('cuda', dtype=torch.float16):
            p = model(x).float().sigmoid()[0, 0].cpu().numpy().astype(np.float16)
        p = np.ascontiguousarray(inv(p, v))
        up = F.interpolate(torch.from_numpy(p.astype(np.float32))[None, None].to(dev), size=(2048, 2048),
                           mode='bilinear', align_corners=False)[0, 0].cpu().numpy()
        out[v] = instances(up, e6)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--prefix', default='e056')
    ap.add_argument('--runs', default='runs/e056-convnext-fold{f}'); args = ap.parse_args()
    import torch
    import segmentation_models_pytorch as smp
    import e006_maskrcnn_refiner as e6, g1_group_eval as g1
    ann_path, images = e6.discover(W / 'data')
    paths = {s: p for s, (split, p) in images.items()}
    test_ids = sorted(s for s, (split, p) in images.items() if split == 'test')
    dev = torch.device('cuda'); torch.backends.cudnn.benchmark = False
    out = W / 'preds/f2_test/_pools'; t0 = time.time(); receipt = {'test_images': len(test_ids), 'folds': {}}
    for f in (0, 1, 2):
        run = W / args.runs.format(f=f)
        ck = torch.load(run / f'e004_fold{f}_unet.pt', map_location='cpu', weights_only=False)
        cfg = ck['config']; size = int(cfg['size'])
        model = smp.Unet(encoder_name=cfg['encoder'], encoder_weights=None, in_channels=1, classes=1)
        model.load_state_dict(ck['state_dict'], strict=True)
        model = model.to(dev).to(memory_format=torch.channels_last).eval()
        ids, _, _, _, opaths = g1.fold_data(f, workers=2)
        cached = {v: pickle.loads((W / f'preds/f1_fusion/pools/{args.prefix}{v}_fold{f}.pkl').read_bytes()) for v in VIEWS}
        worst = 1.0
        with torch.inference_mode():
            for i in range(3):
                pr = predict(model, opaths[ids[i]], size, dev, e6)
                for v in VIEWS:
                    a, b = pr[v], cached[v][i]
                    assert len(a) == len(b), f'fold {f} OOF image {i} view {v}: {len(a)} vs {len(b)} instances'
                    for (ca, sa), (cb, sb) in zip(a, b):
                        da, db = e6.dense(ca), e6.dense(cb); iou = float((da & db).sum() / max((da | db).sum(), 1)); worst = min(worst, iou)
                        assert iou >= 0.99 and abs(sa - sb) < 1e-3, f'fold {f} OOF image {i} view {v}: iou {iou} ds {sa - sb}'
            print(f'fold {f}: OOF reproduction ok (worst IoU {worst:.4f})', flush=True)
            pools = {v: [] for v in VIEWS}
            for i, pid in enumerate(test_ids):
                pr = predict(model, paths[pid], size, dev, e6)
                for v in VIEWS:
                    pools[v].append(pr[v])
        for v in VIEWS:
            (out / f'{args.prefix}_fold{f}_{v or "id"}.pkl').write_bytes(pickle.dumps(pools[v], protocol=4))
        receipt['folds'][f] = {'ckpt_sha': e6.sha_file(run / f'e004_fold{f}_unet.pt'), 'oof_repro_worst_iou': worst,
                               'inst_per_image': {v: float(np.mean([len(x) for x in pools[v]])) for v in VIEWS}}
        print(f'fold {f} test done {receipt["folds"][f]["inst_per_image"]} [{time.time() - t0:.0f}s]', flush=True)
        del model; torch.cuda.empty_cache()
    receipt['seconds'] = time.time() - t0
    receipt['prefix'] = args.prefix; receipt['runs'] = args.runs
    d = W / ('preds/f2_test/e056_d1' if args.prefix == 'e056' else f'preds/f2_test/{args.prefix}'); d.mkdir(parents=True, exist_ok=True)
    (d / 'infer_receipt.json').write_text(json.dumps(receipt, indent=1), encoding='utf-8')


if __name__ == '__main__':
    main()
