"""A2: error analysis of E006 (Mask R-CNN + refiner) OOF predictions, with E004 (U-Net) complementarity.

Classifies every GT instance and every prediction per (image, annotator record) entry, at the
locked setting and at the source default, and stratifies by size, thickness, disk position,
site, year and annotator agreement. Matching = official PQ (IoU > 0.5, unique).

Run (from WORKDIR):  ../../.venv-gpu/Scripts/python.exe code/a2_error_analysis.py [--fold 0]
Outputs preds/a2_error_analysis/fold{F}/: report.json, gt_table.csv, pred_table.csv, instances.pkl
"""
import argparse
import csv
import json
import pickle
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
WORKDIR = HERE.parent
sys.path.insert(0, str(HERE))
import e006_maskrcnn_refiner as e6  # noqa: E402
import pq_metric as pq  # noqa: E402

DATA = WORKDIR / 'data'
LOCKED = (0.8, 0.5, 0.5)
E004_TUNED = (0.6, 5, 256)


def e006_dir(fold):
    return WORKDIR / ('kernels/e006-maskrcnn-refiner-fold0/output/v0' if fold == 0
                      else f'kernels/e008-maskrcnn-refiner-fold{fold}/output/v0')


def e006_instances(kout, fold, ids, paths, device, settings):
    rec = json.loads((kout / 'e006_receipt.json').read_text(encoding='utf-8'))
    ref, _ = e6.load_refiner(kout / rec['checkpoints']['ref']['last']['path'], device)
    raw = dict(np.load(kout / f'e006_fold{fold}_oof_det_last.npz'))
    per = e6.raw_by_image(raw, len(ids))
    out = {s: {} for s in settings}
    for i, pid in enumerate(ids):
        image = e6.load_gray(paths[pid])
        cache = e6.RefinerCache(ref, device, image)
        for s in settings:
            det = e6.detector_masks(per[i], s[0], s[1], e6.Fraction(e6.FULL, int(raw['size'])))
            out[s][pid] = (e6.refined_masks(det, cache, s[2])[0] if det else []) if s[2] is not None else det
    return out


def e004_instances(ids, device):
    """E004 tuned post-processing on the Kaggle fold-0 OOF probabilities -> crops at 2048."""
    import e004_unet as e4
    from scipy import ndimage
    kout = WORKDIR / 'kernels/e004-unet-fold0/output/v0'
    e4_ids = json.loads((kout / 'e004_fold0_oof_ids.json').read_text(encoding='utf-8'))
    probs = np.load(kout / 'e004_fold0_oof_probs.npy', mmap_mode='r')
    idx = {p: i for i, p in enumerate(e4_ids)}
    t, r, a = E004_TUNED
    out = {}
    for pid in ids:
        up = e4.upsample(np.asarray(probs[idx[pid]]), device)
        binary = (up > t).cpu().numpy()
        lab, n = e4.label_instances(binary, r, device)
        lab, n = e4.filter_min_area(lab, n, a)
        crops = []
        for k, sl in enumerate(ndimage.find_objects(lab), 1):
            if sl is not None:
                crops.append((sl[0].start, sl[1].start, lab[sl] == k))
        out[pid] = crops
    return out


def disk_params(path):
    img = e6.resize_input(e6.load_gray(path), 512)
    m = img > 0.1 * img.max()
    ys, xs = np.nonzero(m)
    f = e6.FULL / 512
    return float(ys.mean() * f), float(xs.mean() * f), float(np.sqrt(m.sum() / np.pi) * f)


def inter_matrix(gts, preds):
    """Pixel intersections (int) GT x pred."""
    G, P = len(gts), len(preds)
    inter = np.zeros((G, P), np.int64)
    for i, g in enumerate(gts):
        if not g[3]:
            continue
        for j, p in enumerate(preds):
            inter[i, j] = e6._pair_intersection(g[:3], p)
    return inter


def match(iou):
    """Official PQ: IoU > 0.5 pairs are unique."""
    pairs = np.argwhere(iou > 0.5)
    return {int(i): int(j) for i, j in pairs}


def classify(gts, preds):
    iou = e6.overlap(gts, preds)
    inter = inter_matrix(gts, preds)
    ga = np.array([g[3] for g in gts], np.int64)
    pa = np.array([int(p[2].sum()) for p in preds], np.int64)
    m = match(iou)
    mp = {j: i for i, j in m.items()}
    cov_g = inter / np.maximum(ga[:, None], 1) if len(gts) and len(preds) else np.zeros((len(gts), len(preds)))
    cov_p = inter / np.maximum(pa[None, :], 1) if len(gts) and len(preds) else np.zeros((len(gts), len(preds)))
    spans = (cov_g >= 0.1).sum(0) if len(gts) else np.zeros(len(preds), int)  # GTs each pred touches
    gcls, pcls = [], []
    for i in range(len(gts)):
        if ga[i] == 0:
            gcls.append('empty_gt'); continue
        if i in m:
            gcls.append('tp'); continue
        best = iou[i].max() if len(preds) else 0.0
        big = np.flatnonzero(cov_g[i] >= 0.5) if len(preds) else []
        if any(spans[j] >= 2 for j in big):
            gcls.append('merged')
        elif len(preds) and (cov_g[i] >= 0.1).sum() >= 2:
            gcls.append('fragmented')
        elif best >= 0.1:
            gcls.append('low_iou')
        else:
            gcls.append('missed')
    for j in range(len(preds)):
        if j in mp:
            pcls.append('tp'); continue
        best = iou[:, j].max() if len(gts) else 0.0
        if spans[j] >= 2:
            pcls.append('merged')
        elif len(gts) and cov_p[:, j].max() >= 0.5:
            pcls.append('fragment_or_inside')
        elif best >= 0.1:
            pcls.append('partial')
        else:
            pcls.append('background')
    return iou, gcls, pcls, m


def bin_of(x, edges, labels):
    for e, lab in zip(edges, labels):
        if x < e:
            return lab
    return labels[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--no-e004', action='store_true')
    ap.add_argument('--run-dir', default=None)
    ap.add_argument('--tag', default='')
    ap.add_argument('--v2', action='store_true', help='primary = post-proc v2 (code/s1_candidates.pool, score >= 0.8)')
    args = ap.parse_args()
    t0 = time.time()
    out = WORKDIR / f'preds/a2_error_analysis/fold{args.fold}{args.tag}'
    out.mkdir(parents=True, exist_ok=True)
    kout = Path(args.run_dir) if args.run_dir else e006_dir(args.fold)
    ids = json.loads((kout / f'e006_fold{args.fold}_oof_ids.json').read_text(encoding='utf-8'))
    ann_path, images = e6.discover(DATA)
    paths = {s: p for s, (split, p) in images.items()}
    rec_phys, rec_anns = e6.load_annotations(ann_path, images)
    phys_recs = defaultdict(list)
    for rid, pid in sorted(rec_phys.items()):
        if rid in rec_anns:
            phys_recs[pid].append(rid)
    tasks = [(paths[p], None, [[a['segmentation'] for a in rec_anns[r]] for r in phys_recs[p]]) for p in ids]
    entries = {p: x[3] for p, x in zip(ids, e6.run_pool(e6.prep_task, tasks, args.workers))}
    print(f'[{time.time()-t0:.0f}s] GT prepared: {len(ids)} images, {sum(len(v) for v in entries.values())} entries', flush=True)

    device = torch.device('cuda')
    settings = [LOCKED, e6.DEFAULT]
    inst = e006_instances(kout, args.fold, ids, paths, device, settings)
    models = {'e006_locked': inst[LOCKED], 'e006_default': inst[e6.DEFAULT]}
    if args.v2:  # replace the primary with exact v2 (same refiner): NMS 0.5 + subtraction, score >= 0.8
        import s1_candidates as s1
        rec = json.loads((kout / 'e006_receipt.json').read_text(encoding='utf-8'))
        ref, _ = e6.load_refiner(kout / rec['checkpoints']['ref']['last']['path'], device)
        raw = dict(np.load(kout / f'e006_fold{args.fold}_oof_det_last.npz'))
        per = e6.raw_by_image(raw, len(ids))
        factor = e6.Fraction(e6.FULL, int(raw['size']))
        v2 = {}
        for i, pid in enumerate(ids):
            cache = e6.RefinerCache(ref, device, e6.load_gray(paths[pid]))
            v2[pid] = [c for c, pl in s1.pool(per[i], cache, 0.4, factor) if pl['score'] >= 0.8]
        models['e006_locked'] = v2
    if args.fold == 0 and not args.no_e004:
        models['e004_tuned'] = e004_instances(ids, device)
    print(f'[{time.time()-t0:.0f}s] instances built', flush=True)
    disks = {p: disk_params(paths[p]) for p in ids}

    report = {'fold': args.fold, 'images': len(ids), 'run_dir': str(kout), 'primary': 'v2' if args.v2 else 'locked', 'models': {}}
    gt_rows, pred_rows = [], []
    tp_sets = {}
    for name, preds_by in models.items():
        mats, gc, pc = [], Counter(), Counter()
        strata = defaultdict(Counter)
        tp_keys = set()
        fp_other = Counter()
        for pid in ids:
            preds = preds_by[pid]
            cy, cx, R = disks[pid]
            site, year = pid[-2], pid[:4]
            n_rec = len(entries[pid])
            per_entry = [classify(g, preds) for g in entries[pid]]
            for e_idx, (gts, (iou, gcls, pcls, m)) in enumerate(zip(entries[pid], per_entry)):
                mats.append(iou)
                for i, (g, c) in enumerate(zip(gts, gcls)):
                    gc[c] += 1
                    if c == 'empty_gt':
                        continue
                    y0, x0, crop, area = g
                    h, w = crop.shape
                    thick = area / max(h, w)
                    rr = np.hypot(y0 + h / 2 - cy, x0 + w / 2 - cx) / R
                    agree = None
                    if n_rec > 1:  # fraction of other records with an IoU>0.5 GT
                        hits = 0
                        for o, ogts in enumerate(entries[pid]):
                            ov = [x[:3] for x in ogts if x[3]]
                            if o != e_idx and ov and e6.overlap([g], ov).max() > 0.5:
                                hits += 1
                        agree = hits / (n_rec - 1)
                    keys = {
                        'area': bin_of(area, [2000, 8000, 32000], ['<2k', '2k-8k', '8k-32k', '>=32k']),
                        'thickness': bin_of(thick, [8, 16, 32], ['<8px', '8-16px', '16-32px', '>=32px']),
                        'radius': bin_of(rr, [0.5, 0.8, 0.95], ['<0.5R', '0.5-0.8R', '0.8-0.95R', '>=0.95R']),
                        'site': site, 'year': year,
                        'agreement': 'single_record' if agree is None else bin_of(agree, [1e-9, 0.999], ['none', 'partial', 'all']),
                    }
                    for k, v in keys.items():
                        strata[f'{k}={v}'][c] += 1
                    if c == 'tp':
                        tp_keys.add((pid, e_idx, i))
                    if name == 'e006_locked':
                        gt_rows.append({'image': pid, 'entry': e_idx, 'gt': i, 'class': c, 'area': area, 'thickness': round(thick, 2),
                                        'r_over_R': round(rr, 3), 'site': site, 'year': year,
                                        'agreement': '' if agree is None else round(agree, 3),
                                        'best_iou': round(float(iou[i].max()) if iou.shape[1] else 0.0, 3)})
                for j, c in enumerate(pcls):
                    pc[c] += 1
                    if c != 'tp' and n_rec > 1:  # does another annotator's record match this FP?
                        other = any(len(ogts) and e6.overlap(ogts, [preds[j]]).max() > 0.5
                                    for o, ogts in enumerate(entries[pid]) if o != e_idx)
                        fp_other[c + ('_matched_by_other_record' if other else '_unmatched_everywhere')] += 1
                    if name == 'e006_locked':
                        y0, x0, msk = preds[j]
                        pred_rows.append({'image': pid, 'entry': e_idx, 'pred': j, 'class': c, 'area': int(msk.sum()),
                                          'site': site, 'n_records': n_rec})
        res = pq.score_matrices(mats)
        tp_sets[name] = tp_keys
        report['models'][name] = {'pq': res, 'gt_classes': dict(gc), 'pred_classes': dict(pc),
                                  'fp_vs_other_records': dict(fp_other),
                                  'strata': {k: dict(v) for k, v in sorted(strata.items())}}
        print(f'[{time.time()-t0:.0f}s] {name}: PQ {res["pq"]:.4f} gt {dict(gc)} pred {dict(pc)}', flush=True)

    if 'e004_tuned' in tp_sets:
        a, b = tp_sets['e006_locked'], tp_sets['e004_tuned']
        report['complementarity'] = {'tp_e006_only': len(a - b), 'tp_e004_only': len(b - a), 'tp_both': len(a & b),
                                     'gt_total': sum(v for k, v in report['models']['e006_locked']['gt_classes'].items() if k != 'empty_gt')}
    for fn, rows in (('gt_table.csv', gt_rows), ('pred_table.csv', pred_rows)):
        with open(out / fn, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    with open(out / 'instances.pkl', 'wb') as f:
        pickle.dump({'ids': ids, 'models': models, 'disks': disks}, f, protocol=4)
    report['seconds'] = time.time() - t0
    (out / 'report.json').write_text(json.dumps(report, indent=1, default=str), encoding='utf-8')
    print(json.dumps({k: report[k] for k in report if k not in ('models',)}, indent=1, default=str))


if __name__ == '__main__':
    main()
