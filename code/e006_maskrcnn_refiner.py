"""E006: private validation-mode replication of nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner.

Our own rewrite (plan.md "Replication review: three public notebooks": verdict replicate with changes).
Nothing from the notebook is executed; only the reviewed method is re-implemented here.

Source pipeline kept: Mask R-CNN ResNet50-FPN v2 (torchvision COCO weights) on 1024 px grayscale->3ch input,
2-class heads, anchors 16..256 x ratios (0.5, 1, 2), 28 px mask RoI pool, SGD lr 0.005 / momentum 0.9 /
wd 1e-4, MultiStep x0.1 at 70%/90% of 10 epochs, AMP, batch 1 per GPU, DDP over 2 GPUs, seed 2026(+rank);
detector masks: score >= 0.60, mask prob >= 0.50, merge pixel-overlapping masks, nearest upsample to 2048;
refiner: ResNet18(ImageNet)-U-Net on 256 px square crops (context 1.8, min side 96), 8 epochs, batch 16,
AdamW 3e-4 / wd 1e-4, cosine, 0.5 BCE + 0.5 soft Dice, AMP; refined mask = refiner prob >= 0.85 in the
crop window (fallback: detector mask if empty), then a second overlap merge.

Required changes (plan.md, C1-C4):
  C1 frozen folds.csv: train on folds 1-2, validate on fold 0 (all annotator records of a physical image stay
     together). Replaces the source's train_test_split on one record per image.
  C2 thresholds calibrated on official PQ semantics (code/pq_metric.py) over every annotator entry, at 2048;
     fixed-default PQ (source defaults) and tuned PQ (selection-biased) are reported separately.
  C3 internet only for the torchvision weight download (COCO Mask R-CNN v2 + ImageNet ResNet18); URLs,
     file hashes and licenses go into the receipt.
  C4 validation mode only: no final retrain on all data, no test inference, no submission.
Every other deviation is listed in DEVIATIONS (also written to the receipt).

Order: weights -> detector DDP training (first-epoch timing check, early stop) -> checkpoints -> CPU crop/GT
cache -> refiner training -> checkpoints -> raw detector OOF saved -> fixed-default PQ -> validator
round-trip -> time-bounded PQ grid (tuned = selection-biased).

Kaggle: python e006_maskrcnn_refiner.py   (writes to /kaggle/working)
Local CPU smoke: see code/BUILDER_NOTES or the E006 report (--image-size 256 --det-epochs 1 --max-steps 2 ...).
"""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from fractions import Fraction
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

import numpy as np

T0 = time.monotonic()
T0_WALL = time.time()
EMBEDDED = {}  # filled by code/build_e006_kernel.py: {filename: source text}
EMBEDDED_FOLDS_SHA256 = None
SOURCE_REF = 'nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner'
SOURCE_SNAPSHOT_SHA256 = None  # filled by the builder (snapshots/notebooks/.../v0.ipynb)

# ---- source hyper-parameters (unchanged)
SEED = 2026
IMAGE_SIZE = 1024
SCORE_THRESHOLD, MASK_THRESHOLD, REFINER_THRESHOLD = 0.60, 0.50, 0.85
REFINER_VAL_THRESHOLD = 0.65
DET_EPOCHS, DET_LR, DET_MOMENTUM, DET_WD = 10, 0.005, 0.9, 1e-4
ANCHOR_SIZES = ((16,), (32,), (64,), (128,), (256,))
ANCHOR_RATIOS = ((0.5, 1.0, 2.0),) * 5
CROP, CONTEXT, MIN_SIDE = 256, 1.8, 96
REF_EPOCHS, REF_BATCH, REF_VAL_BATCH, REF_LR, REF_WD = 8, 16, 32, 3e-4, 1e-4
REF_INFER_BATCH = 32
# ---- calibration grid: source score x mask grid + refiner thresholds (source fixed 0.85); None = no refiner
GRID_SCORE = (0.40, 0.50, 0.60, 0.70, 0.80)
GRID_MASK = (0.40, 0.50, 0.60)  # stored exactly as 3 levels (see detector_raw)
GRID_REF = (None, 0.50, 0.65, 0.75, 0.85, 0.90)
DEFAULT = (SCORE_THRESHOLD, MASK_THRESHOLD, REFINER_THRESHOLD)
DEFAULT_DET_ONLY = (SCORE_THRESHOLD, MASK_THRESHOLD, None)
FULL = 2048

DEVIATIONS = [
    'C1 split: source = sklearn train_test_split 80/20 (site x count-bin strata, seed 2026) on one record per '
    'physical image; here frozen folds.csv, train folds 1-2, validate fold 0.',
    'C2 calibration: source picks score/mask thresholds by Hungarian mean Dice at 1024 (selected + all annotator '
    'sets) and decides use_refiner by Dice; here official PQ at 2048 over every annotator entry. Grid adds refiner '
    'thresholds {off,0.5,0.65,0.75,0.85,0.9} to the source score x mask grid. Tuned PQ is selection-biased.',
    'C3 internet ON only for torchvision weight download (same weights as source: MaskRCNN_ResNet50_FPN_V2 '
    'DEFAULT, ResNet18 DEFAULT); URLs/hashes/licenses recorded.',
    'C4 validation mode only: no final retrain on all data (source cells 25-27), no test inference/submission.',
    'Training record per physical image: kept the source rule (record with most filaments, then largest summed '
    'annotation area, then largest id string). Same rule picks the fold-0 record for the source-style in-training '
    'val metric and refiner val crops. PQ uses every annotator record.',
    'Checkpoint selection: source keeps the best epoch by its fold-0 val metric (detector: mean Dice; refiner: '
    'crop Dice @0.65). We save last and best. PRIMARY = last detector + last refiner (no selection on fold 0); '
    'SOURCE-FAITHFUL = best + best, reported as selection-biased.',
    'Time guards added: first-epoch projection abort, per-epoch early stop, watchdog, time-bounded grid. If early '
    'stop fires, fewer than 10 detector epochs are trained (recorded).',
    'Implementation: masks handled as bbox crops (exact): overlap merge = connected groups of pixel-overlapping '
    'masks (source: RLE IoU > 0); nearest upsample = pixel repeat (cv2 INTER_NEAREST at integer factor); PQ IoU = '
    'pq_metric float32 semantics. Unit-tested against dense references.',
    'Raw detector OOF cached for score >= 0.40 with mask probs stored as 3 exact levels (>=0.4/0.5/0.6, fp32 '
    'compare on device); source caches the same grid in memory.',
    'Empty rasterised annotations are skipped for refiner crops (source square_bounds would fail on them); they '
    'still count as GT instances in PQ.',
    'Images: only physical images with annotation rows (same set as E004); every record has >= 1 instance.',
    'Single script; DDP via python -m torch.distributed.run (source: %%writefile + !torchrun). NCCL timeout 60 min.',
]


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def elapsed():
    """Seconds since the orchestrator started (wall clock, shared with the DDP ranks)."""
    return time.time() - T0_WALL


LOG_PATH = None


def log(msg):
    line = f'[{elapsed():8.1f}s] {msg}'
    print(line, flush=True)
    if LOG_PATH is not None:
        with open(LOG_PATH, 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def dump(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=True), encoding='utf-8')
    tmp.replace(path)


def setup_lib(out):
    if EMBEDDED:
        lib = out / 'lib'
        lib.mkdir(parents=True, exist_ok=True)
        for name, text in EMBEDDED.items():
            (lib / name).write_text(text, encoding='utf-8')
        sys.path.insert(0, str(lib))
        return {n: sha_bytes(t.encode('utf-8')) for n, t in EMBEDDED.items()}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return {}


def versions():
    out = {}
    for p in ('torch', 'torchvision', 'numpy', 'opencv-python', 'opencv-python-headless', 'pycocotools', 'scipy'):
        try:
            out[p] = importlib.metadata.version(p)
        except importlib.metadata.PackageNotFoundError:
            out[p] = None
    out['python'] = sys.version.split()[0]
    return out


# ------------------------------------------------------------------ data
def discover(inp):
    anns = sorted(p for p in inp.rglob('*.json') if 'train' in p.name.lower() and 'annotation' in p.name.lower())
    if len(anns) != 1:
        raise ValueError(f'Expected one train annotation JSON, found {anns}')
    images = {}
    for p in sorted(inp.rglob('*')):
        if p.suffix.lower() in ('.jpg', '.jpeg'):
            parts = {x.lower() for x in p.parts}
            split = 'train' if 'train_images' in parts else 'test' if 'test_images' in parts else None
            if split:
                if p.stem in images:
                    raise ValueError(f'duplicate image stem {p.stem}')
                images[p.stem] = (split, p)
    return anns[0], images


def load_annotations(ann_path, images):
    """-> rec_phys {record_id: physical_id}, rec_anns {record_id: [annotation dicts]} (train only)."""
    data = json.loads(Path(ann_path).read_text(encoding='utf-8'))
    by_name = {p.name: s for s, (split, p) in images.items() if split == 'train'}
    rec_phys = {}
    for r in data['images']:
        name = Path(r['file_name']).name
        if name not in by_name:
            raise ValueError(f'record {r["id"]} has unresolved file {name}')
        rec_phys[str(r['id'])] = by_name[name]
    rec_anns = {}
    for a in data['annotations']:
        rec_anns.setdefault(str(a['image_id']), []).append(a)
    return rec_phys, rec_anns


def select_record(rids, rec_anns):
    """Source rule (cell 17): sort by (n_filaments, total annotation 'area', id string), keep the last."""
    return max(rids, key=lambda r: (len(rec_anns[r]), sum(float(a['area']) for a in rec_anns[r]), str(r)))


def read_folds(args):
    if args.folds_csv:
        text = Path(args.folds_csv).read_text(encoding='utf-8')
    else:
        text = EMBEDDED['folds.csv']
        if EMBEDDED_FOLDS_SHA256 and sha_bytes(text.encode('utf-8')) != EMBEDDED_FOLDS_SHA256:
            raise ValueError('embedded folds.csv hash mismatch')
    rows = list(csv.DictReader(text.splitlines()))
    return {r['record_id']: (r['physical_id'], int(r['fold'])) for r in rows}, sha_bytes(text.encode('utf-8'))


def seg_to_mask(seg, size=FULL):
    """coco.annToMask equivalent for a 2048x2048 record."""
    from pycocotools import mask as cm
    if isinstance(seg, list):
        rle = cm.merge(cm.frPyObjects(seg, size, size))
    elif isinstance(seg.get('counts'), list):
        rle = cm.frPyObjects(seg, size, size)
    else:
        rle = {'size': seg['size'], 'counts': seg['counts'].encode('ascii') if isinstance(seg['counts'], str) else seg['counts']}
    return cm.decode(rle)


def load_gray(path):
    import cv2
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None or img.shape != (FULL, FULL):
        raise ValueError(f'{path}: unreadable or not 2048x2048')
    return img


def resize_input(img, size):
    import cv2
    import numpy as np
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255


# ------------------------------------------------------------------ crop-mask ops (exact, unit-tested)
# A mask is (y0, x0, bool array) in some frame; arrays are trimmed to their nonzero bbox.
def trim(y0, x0, arr):
    import numpy as np
    rows, cols = np.flatnonzero(arr.any(1)), np.flatnonzero(arr.any(0))
    if not len(rows):
        return None
    return (y0 + int(rows[0]), x0 + int(cols[0]), np.ascontiguousarray(arr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1], dtype=bool))


def _pair_intersection(a, b):
    ay, ax, am = a
    by, bx, bm = b
    y0, y1 = max(ay, by), min(ay + am.shape[0], by + bm.shape[0])
    x0, x1 = max(ax, bx), min(ax + am.shape[1], bx + bm.shape[1])
    if y0 >= y1 or x0 >= x1:
        return 0
    return int(np.count_nonzero(am[y0 - ay:y1 - ay, x0 - ax:x1 - ax] & bm[y0 - by:y1 - by, x0 - bx:x1 - bx]))


def merge_crops(crops):
    """Source merge_overlaps: connected groups of masks with any shared pixel -> OR. Group order = first index."""
    crops = [c for c in crops if c is not None]
    n = len(crops)
    if n < 2:
        return crops
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for i in range(n):
        for j in range(i + 1, n):
            if find(i) != find(j) and _pair_intersection(crops[i], crops[j]):
                parent[find(j)] = find(i)
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for g in sorted(groups.values(), key=min):
        if len(g) == 1:
            out.append(crops[g[0]])
            continue
        y0 = min(crops[i][0] for i in g)
        x0 = min(crops[i][1] for i in g)
        y1 = max(crops[i][0] + crops[i][2].shape[0] for i in g)
        x1 = max(crops[i][1] + crops[i][2].shape[1] for i in g)
        u = np.zeros((y1 - y0, x1 - x0), bool)
        for i in g:
            cy, cx, cm_ = crops[i]
            u[cy - y0:cy - y0 + cm_.shape[0], cx - x0:cx - x0 + cm_.shape[1]] |= cm_
        out.append((y0, x0, u))
    return out


def upscale(crop, f):
    """Nearest upsample of a crop from detector size to FULL. f = FULL / size (int or Fraction).
    Integer f: pixel repeat (== cv2 INTER_NEAREST). Non-integer f (e.g. 1536 -> 4/3): full pixel Y takes source
    row floor(Y / f), computed with exact integer arithmetic."""
    y0, x0, m = crop
    f = Fraction(f)
    if f == 1:
        return crop
    if f.denominator == 1:
        k = int(f)
        return (y0 * k, x0 * k, np.repeat(np.repeat(m, k, 0), k, 1))
    num, den = f.numerator, f.denominator

    def axis(a0, n):
        lo, hi = -((-a0 * num) // den), -((-(a0 + n) * num) // den)  # ceil
        return lo, (np.arange(lo, hi) * den) // num - a0

    ya, iy = axis(y0, m.shape[0])
    xa, ix = axis(x0, m.shape[1])
    return (ya, xa, m[np.ix_(iy, ix)])


def square_bounds(crop, height=FULL, width=FULL, context=CONTEXT, minimum=MIN_SIDE):
    """Source square_bounds(mask) computed from the trimmed bbox (identical: bbox is all it uses)."""
    y0m, x0m, m = crop
    xmin, xmax1, ymin, ymax1 = x0m, x0m + m.shape[1], y0m, y0m + m.shape[0]
    center_x, center_y = (xmin + xmax1) / 2, (ymin + ymax1) / 2
    side = int(math.ceil(max(xmax1 - xmin, ymax1 - ymin) * context))
    side = min(max(side, minimum), min(height, width))
    x0, y0 = int(round(center_x - side / 2)), int(round(center_y - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0; x0 = 0
    if y0 < 0:
        y1 -= y0; y0 = 0
    if x1 > width:
        x0 -= x1 - width; x1 = width
    if y1 > height:
        y0 -= y1 - height; y1 = height
    return int(x0), int(y0), int(x1), int(y1)


def dense(crop, size=FULL):
    out = np.zeros((size, size), np.uint8)
    if crop is not None:
        y0, x0, m = crop
        out[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
    return out


def gt_crop(mask):
    """GT instance at 2048 -> (y0, x0, bool crop, area); empty -> area 0 (kept: official stack count)."""
    c = trim(0, 0, mask.astype(bool))
    if c is None:
        return (0, 0, np.zeros((0, 0), bool), 0)
    return (c[0], c[1], c[2], int(c[2].sum()))


def overlap(gts, preds):
    """IoU matrix with pq_metric.overlap_matrix float32 semantics for crop GT (with area) vs crop preds."""
    G, P = len(gts), len(preds)
    inter = np.zeros((G, P), np.int64)
    ga = np.array([g[3] for g in gts], np.int64)
    pa = np.array([int(p[2].sum()) for p in preds], np.int64)
    if G and P:
        gb = np.array([(g[0], g[1], g[0] + g[2].shape[0], g[1] + g[2].shape[1]) for g in gts])
        pb = np.array([(p[0], p[1], p[0] + p[2].shape[0], p[1] + p[2].shape[1]) for p in preds])
        hit = ((np.maximum(gb[:, None, 0], pb[None, :, 0]) < np.minimum(gb[:, None, 2], pb[None, :, 2])) &
               (np.maximum(gb[:, None, 1], pb[None, :, 1]) < np.minimum(gb[:, None, 3], pb[None, :, 3])) &
               (ga[:, None] > 0))
        for i, j in zip(*np.nonzero(hit)):
            inter[i, j] = _pair_intersection(gts[i][:3], preds[j])
    i32 = inter.astype(np.float32)
    union = ga.astype(np.float32)[:, None] + pa.astype(np.float32)[None, :] - i32
    return np.divide(i32, union, out=np.zeros_like(i32), where=union != 0)


# ------------------------------------------------------------------ models (source architecture)
def build_detector(pretrained, image_size):
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2, MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.ops import MultiScaleRoIAlign
    weights = MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None
    model = maskrcnn_resnet50_fpn_v2(weights=weights, weights_backbone=None)
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, hidden, 2)
    model.rpn.anchor_generator = AnchorGenerator(sizes=ANCHOR_SIZES, aspect_ratios=ANCHOR_RATIOS)
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(featmap_names=['0', '1', '2', '3'], output_size=28, sampling_ratio=2)
    model.transform.min_size, model.transform.max_size = (image_size,), image_size
    return model


def build_refiner(pretrained):
    import torch
    from torch import nn
    from torchvision.models import resnet18, ResNet18_Weights

    class ConvBlock(nn.Sequential):
        def __init__(self, i, o):
            super().__init__(nn.Conv2d(i, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True),
                             nn.Conv2d(o, o, 3, padding=1, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))

    class CropUNet(nn.Module):
        def __init__(self):
            super().__init__()
            enc = resnet18(weights=ResNet18_Weights.DEFAULT if pretrained else None)
            self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
            self.pool, self.layer1, self.layer2, self.layer3, self.layer4 = enc.maxpool, enc.layer1, enc.layer2, enc.layer3, enc.layer4
            self.up4, self.dec4 = nn.ConvTranspose2d(512, 256, 2, 2), ConvBlock(512, 256)
            self.up3, self.dec3 = nn.ConvTranspose2d(256, 128, 2, 2), ConvBlock(256, 128)
            self.up2, self.dec2 = nn.ConvTranspose2d(128, 64, 2, 2), ConvBlock(128, 64)
            self.up1, self.dec1 = nn.ConvTranspose2d(64, 32, 2, 2), ConvBlock(96, 32)
            self.up0, self.head = nn.ConvTranspose2d(32, 16, 2, 2), nn.Conv2d(16, 1, 1)
            self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
            self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        def forward(self, x):
            x = (x - self.mean) / self.std
            x0 = self.stem(x)
            x1 = self.layer1(self.pool(x0))
            x2 = self.layer2(x1)
            x3 = self.layer3(x2)
            x4 = self.layer4(x3)
            x = self.dec4(torch.cat([self.up4(x4), x3], 1))
            x = self.dec3(torch.cat([self.up3(x), x2], 1))
            x = self.dec2(torch.cat([self.up2(x), x1], 1))
            x = self.dec1(torch.cat([self.up1(x), x0], 1))
            return self.head(self.up0(x))
    return CropUNet()


# ------------------------------------------------------------------ detector stage (runs inside each DDP rank)
def seed_everything(seed):
    import torch
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    import torch
    seed = torch.initial_seed() % 2 ** 32
    random.seed(seed); np.random.seed(seed)


def photometric(image):
    """Source FilamentDataset.photometric (applied on the disk region > 0.01)."""
    import cv2
    valid = image > 0.01
    if np.random.rand() < 0.70:
        image[valid] = image[valid] ** np.random.uniform(0.80, 1.25)
    if np.random.rand() < 0.70:
        mean = image[valid].mean()
        contrast, gain = np.random.uniform(0.85, 1.15), np.random.uniform(0.90, 1.10)
        image[valid] = ((image[valid] - mean) * contrast + mean) * gain
    if np.random.rand() < 0.15:
        blurred = cv2.GaussianBlur(image, (0, 0), np.random.uniform(0.30, 0.80))
        image[valid] = blurred[valid]
    if np.random.rand() < 0.25:
        image[valid] += np.random.normal(0, np.random.uniform(0.003, 0.015), valid.sum())
    return np.clip(image, 0, 1)



INPUT_MODE = 'gray'  # 'gray' = grey image x 3 (source); 'phys' = grey, limb-flattened, black top-hat (E034)


def phys_channels(img):
    """E034 input: (grey, limb-darkening-flattened, multi-scale black top-hat) from the image alone. img float [0,1] HxW."""
    import cv2
    S = img.shape[0]
    small = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
    centre = small[64:192, 64:192]
    disk = (small > 0.6 * float(np.median(centre))).astype(np.uint8)  # solar disk, excludes the faint outer halo
    n, lab, st, _ = cv2.connectedComponentsWithStats(disk, connectivity=8)
    if n <= 1 or st[1:, cv2.CC_STAT_AREA].max() < 1000:
        return np.stack([img, img, np.zeros_like(img)])
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    disk = lab == k
    ys, xs = np.nonzero(disk)
    cy, cx, R = ys.mean(), xs.mean(), max(np.sqrt(disk.sum() / np.pi), 1.0)
    yy, xx = np.mgrid[:256, :256]
    rb = np.clip((np.hypot(yy - cy, xx - cx) / R * 50).astype(int), 0, 59)
    prof = np.full(60, np.nan)
    for b in range(60):
        v = small[(rb == b) & disk]
        if v.size > 20:
            prof[b] = np.median(v)
    valid = np.flatnonzero(~np.isnan(prof))
    prof = np.interp(np.arange(60), valid, prof[valid])  # fill gaps (centre bin, beyond the limb) from neighbours
    f = S / 256.0
    ry = (np.arange(S, dtype=np.float32) + 0.5) / f - 0.5 - cy
    rx = (np.arange(S, dtype=np.float32) + 0.5) / f - 0.5 - cx
    r = np.sqrt(ry[:, None] ** 2 + rx[None, :] ** 2) / R
    bg = np.interp(r * 50, np.arange(60) + 0.5, prof).astype(np.float32)
    inside = r <= 1.0
    flat = np.where(inside, img / np.maximum(bg, 1e-3), 1.0)
    ch2 = ((np.clip(flat, 0.5, 1.2) - 0.5) / 0.7).astype(np.float32)
    sm = cv2.GaussianBlur(ch2, (0, 0), max(0.5, 1.5 * S / FULL))  # suppress granulation before the top-hat
    bh = np.zeros_like(ch2)
    for rad in (5, 11, 21):
        kk = max(1, int(round(rad * S / FULL)))
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * kk + 1, 2 * kk + 1))
        bh = np.maximum(bh, cv2.morphologyEx(sm, cv2.MORPH_BLACKHAT, ker))
    ch3 = np.where(r <= 0.98, np.clip(bh * 3.0, 0, 1), 0).astype(np.float32)
    return np.stack([img.astype(np.float32), ch2, ch3])


def det_input(img):
    """Detector input (3, H, W) float32 from the resized grey image; used by training, OOF, TTA and test."""
    img = np.ascontiguousarray(img, dtype=np.float32)
    if INPUT_MODE == 'phys':
        return phys_channels(img)
    return np.stack([img, img, img])


def union_record(task):
    """E032 union ground truth (plan v3 E032). task = (pid, [[seg] per record, ordered, anchor first]).
    -> (pid, [compressed RLE dicts], audit dict)."""
    from pycocotools import mask as cm
    from scipy.optimize import linear_sum_assignment
    import cv2
    pid, recs = task
    inst = []
    for segs in recs:
        lst = []
        for j, seg in enumerate(segs):
            c = trim(0, 0, seg_to_mask(seg).astype(bool))
            if c is not None:
                lst.append((j, c))
        inst.append(lst)

    def iou(a, b):
        i = _pair_intersection(a, b)
        return i / max(1, int(a[2].sum()) + int(b[2].sum()) - i)
    clusters = [[(0, j, c)] for j, c in inst[0]]  # members: (record order, instance index, crop)
    skipped = 0
    for ri in range(1, len(inst)):
        cand = inst[ri]
        if not cand:
            continue
        pairs = {}
        if clusters:
            M = np.zeros((len(cand), len(clusters)))
            for a, (_, c) in enumerate(cand):
                for b, cl in enumerate(clusters):
                    v = iou(c, cl[0][2])
                    if v > 0.3:
                        M[a, b] = 1000.0 + v  # max number of matches first, then total IoU
            ra, cb = linear_sum_assignment(-M)
            pairs = {int(a): int(b) for a, b in zip(ra, cb) if M[a, b] > 0}
        for a, (j, c) in enumerate(cand):
            if a in pairs:
                clusters[pairs[a]].append((ri, j, c))
                continue
            ac = int(c[2].sum())  # granularity variant of an existing cluster (piece of it / superset): not a new filament
            if any(_pair_intersection(c, cl[0][2]) / max(1, min(ac, int(cl[0][2][2].sum()))) >= 0.5 for cl in clusters):
                skipped += 1
                continue
            clusters.append([(ri, j, c)])
    masks, prio = [], []
    for cl in clusters:
        recs_in = [m[0] for m in cl]
        if len(set(recs_in)) != len(recs_in):
            raise RuntimeError(f'union {pid}: a cluster holds two instances of one record')
        y0 = min(m[2][0] for m in cl); x0 = min(m[2][1] for m in cl)
        y1 = max(m[2][0] + m[2][2].shape[0] for m in cl); x1 = max(m[2][1] + m[2][2].shape[1] for m in cl)
        cnt = np.zeros((y1 - y0, x1 - x0), np.int32)
        first = np.zeros_like(cnt, bool)
        for k, (_, _, (cy, cx, cmk)) in enumerate(cl):
            cnt[cy - y0:cy - y0 + cmk.shape[0], cx - x0:cx - x0 + cmk.shape[1]] += cmk
            if k == 0:
                first[cy - y0:cy - y0 + cmk.shape[0], cx - x0:cx - x0 + cmk.shape[1]] = cmk
        n = len(cl)
        keep = cnt * 2 > n
        if n % 2 == 0:
            keep |= (cnt * 2 == n) & first
        if not keep.any():
            keep = first
        masks.append((y0, x0, keep))
        prio.append((-n, cl[0][0], cl[0][1]))
    cover = np.zeros((FULL, FULL), np.uint8)
    for y0, x0, m in masks:
        sub = cover[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        sub += (m & (sub < 2)).astype(np.uint8)
    disputed, total = int((cover >= 2).sum()), int((cover >= 1).sum())
    taken = np.zeros((FULL, FULL), bool)
    out, removed = [], 0
    for idx in sorted(range(len(masks)), key=lambda i: prio[i]):
        y0, x0, m = masks[idx]
        m = m & ~taken[y0:y0 + m.shape[0], x0:x0 + m.shape[1]]
        nl, lab, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
        for k in range(1, nl):
            if st[k, cv2.CC_STAT_AREA] < 16:
                m[lab == k] = False
        if not m.any():
            removed += 1
            continue
        taken[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] |= m
        full = np.zeros((FULL, FULL), np.uint8)
        full[y0:y0 + m.shape[0], x0:x0 + m.shape[1]] = m
        rle = cm.encode(np.asfortranarray(full))
        out.append({'size': [FULL, FULL], 'counts': rle['counts'].decode('ascii')})
    sizes = [len(c) for c in clusters]
    member_iou = [iou(masks[i], m[2]) for i, cl in enumerate(clusters) for m in cl]
    audit = {'records': len(recs), 'selected_instances': len(inst[0]), 'clusters': len(clusters), 'skipped_granularity': skipped,
             'single_member': sum(z == 1 for z in sizes), 'out_instances': len(out), 'removed_empty': removed,
             'disputed_px': disputed, 'union_px': total, 'disputed_share': disputed / max(1, total),
             'mean_member_iou': float(np.mean(member_iou)) if member_iou else None}
    return pid, out, audit


class HardMixSampler:
    """E070 hard-instance expert: per epoch len(dataset) samples, (1 - frac) drawn uniformly without replacement from all
    training images and frac uniformly with replacement from the hard images, shuffled by a (seed, epoch) generator,
    padded to a multiple of the world size and split over ranks like DistributedSampler (same epoch length)."""

    def __init__(self, n, hard, frac, world, rank, seed):
        self.n, self.hard, self.frac, self.world, self.rank, self.seed = n, np.asarray(hard, np.int64), frac, world, rank, seed
        self.epoch = 0
        self.per_rank = -(-n // world)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def indices(self):
        rng = np.random.default_rng([self.seed, self.epoch])
        k = int(round(self.n * self.frac))
        idx = np.concatenate([rng.choice(self.n, self.n - k, replace=False), rng.choice(self.hard, k, replace=True)])
        rng.shuffle(idx)
        idx = np.concatenate([idx, idx[:self.per_rank * self.world - len(idx)]])
        return idx[self.rank::self.world]

    def __iter__(self):
        return iter(self.indices().tolist())

    def __len__(self):
        return self.per_rank


def hard_items(items, segs, area_max, thick_max, quantile=0.75):
    """E070: per training item, the number of instances of its selected record with full-resolution area < area_max or
    thickness (area / max(bbox h, w)) < thick_max (E063 strata); hard = items whose count is >= the `quantile` quantile of
    the training items' counts (and >= 1). Returns (hard indices, counts). (An "any small / thin instance" rule marks 93 %
    of the records and would leave the sampler uniform; amendment before any run, decisions.md.)"""
    counts = []
    for k, (pid, path, rid) in enumerate(items):
        assert not isinstance(rid, list), 'hard sampling needs one selected record per image'
        c = 0
        for seg in segs.get(rid, []):
            m = seg_to_mask(seg)
            a = int(m.sum())
            if not a:
                continue
            rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
            c += int(a < area_max or a / max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1) < thick_max)
        counts.append(c)
    cut = max(1.0, float(np.quantile(counts, quantile)))
    return [k for k, c in enumerate(counts) if c >= cut], counts


class DetDataset:
    """Source FilamentDataset: one (selected) annotator record per physical image."""

    def __init__(self, items, segs, size, augment, input_mode='gray'):
        self.items, self.segs, self.size, self.augment = items, segs, size, augment
        self.input_mode = input_mode  # carried into DataLoader workers (Windows spawn does not inherit the global)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        import cv2
        import torch
        global INPUT_MODE
        INPUT_MODE = self.input_mode
        pid, path, rid = self.items[index]
        if isinstance(rid, list):  # B1: one annotator record drawn at random per sample
            rid = rid[np.random.randint(len(rid))]
        image = resize_input(load_gray(path), self.size)
        masks = []
        for seg in self.segs[rid]:
            m = cv2.resize(seg_to_mask(seg), (self.size, self.size), interpolation=cv2.INTER_NEAREST).astype(bool)
            if np.any(m):
                masks.append(m)
        masks = np.stack(masks) if masks else np.zeros((0, self.size, self.size), bool)
        if self.augment:
            k = np.random.randint(4)
            image, masks = np.rot90(image, k), np.rot90(masks, k, axes=(1, 2))
            if np.random.rand() < 0.5:
                image, masks = np.flip(image, axis=1), np.flip(masks, axis=2)
            image = photometric(np.ascontiguousarray(image))
        image, masks = np.ascontiguousarray(image), np.ascontiguousarray(masks)
        boxes = []
        for m in masks:
            ys, xs = np.where(m)
            boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
        boxes = np.asarray(boxes, np.float32).reshape(-1, 4)
        target = {'boxes': torch.as_tensor(boxes), 'labels': torch.ones(len(masks), dtype=torch.int64),
                  'masks': torch.as_tensor(masks, dtype=torch.uint8), 'image_id': torch.tensor([index]),
                  'area': torch.as_tensor(masks.sum((1, 2)), dtype=torch.float32),
                  'iscrowd': torch.zeros(len(masks), dtype=torch.int64)}
        return torch.from_numpy(det_input(image)), target, pid


def collate_fn(batch):
    return tuple(zip(*batch))


def source_instance_metrics(pred, true):
    """Source instance_metrics (mean Hungarian Dice, F1@0.5, union Dice) on dense bool stacks."""
    from pycocotools import mask as cm
    from scipy.optimize import linear_sum_assignment
    enc = lambda m: cm.encode(np.asfortranarray(m.astype(np.uint8)))  # noqa: E731
    p, t = len(pred), len(true)
    if p and t:
        iou = cm.iou([enc(m) for m in pred], [enc(m) for m in true], [0] * t)
        dice = 2 * iou / (1 + iou + 1e-7)
        r, c = linear_sum_assignment(-dice)
        mean_dice, tp = float(dice[r, c].sum() / max(p, t)), int((iou[r, c] >= 0.5).sum())
    else:
        mean_dice, tp = float(p == 0 and t == 0), 0
    if p or t:
        shape = pred[0].shape if p else true[0].shape
        pu = np.logical_or.reduce(pred) if p else np.zeros(shape, bool)
        tu = np.logical_or.reduce(true) if t else np.zeros(shape, bool)
        union_dice = float(2 * np.logical_and(pu, tu).sum() / (pu.sum() + tu.sum() + 1e-7))
    else:
        union_dice = 1.0
    prec, rec = (tp / p if p else 0.0), (tp / t if t else 0.0)
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {'mean_dice': mean_dice, 'union_dice': union_dice, 'f1_50': f1, 'count_error': abs(p - t), 'predictions': p}


def source_val(model, loader, device, size):
    import torch
    model.eval()
    rows = []
    with torch.inference_mode():
        for images, targets, _ in loader:
            out = model([images[0].to(device, non_blocking=True)])[0]
            keep = out['scores'] >= SCORE_THRESHOLD
            pm = (out['masks'][keep, 0] >= MASK_THRESHOLD).cpu().numpy()
            merged = merge_crops([trim(0, 0, m) for m in pm])
            pred = np.stack([dense(c, size).astype(bool) for c in merged]) if merged else np.zeros((0, size, size), bool)
            rows.append(source_instance_metrics(pred, targets[0]['masks'].numpy().astype(bool)))
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}


def detector_worker(cfg_path):
    """One DDP rank (or the only process when not launched by torch.distributed.run)."""
    import datetime
    import torch
    import torch.distributed as dist
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    import cv2
    global T0_WALL, LOG_PATH, SEED, INPUT_MODE
    cfg = json.loads(Path(cfg_path).read_text(encoding='utf-8'))
    INPUT_MODE = cfg.get('input_mode', 'gray')
    T0_WALL = cfg['t0_wall']
    SEED = cfg.get('seed', SEED)
    accum = max(1, int(cfg.get('accum', 1)))
    cv2.setNumThreads(0)
    distributed = 'LOCAL_RANK' in os.environ
    use_cuda = cfg['device'] == 'cuda'
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')
    if distributed:
        dist.init_process_group('nccl' if use_cuda else 'gloo', timeout=datetime.timedelta(minutes=60))
    rank, world = (dist.get_rank(), dist.get_world_size()) if distributed else (0, 1)
    out = Path(cfg['out'])
    if rank == 0:
        LOG_PATH = out / 'train.log'
    seed_everything(SEED + rank)
    torch.backends.cudnn.benchmark = cfg.get('cudnn_benchmark', 'on') == 'on'  # off: variable RoI counts re-tune every step
    torch.set_float32_matmul_precision('high')
    size = cfg['image_size']

    data = json.loads(Path(cfg['ann_path']).read_text(encoding='utf-8'))
    need = {r for it in cfg['train_items'] + cfg['val_items'] for r in (it[2] if isinstance(it[2], list) else [it[2]])}
    segs = {}
    for a in data['annotations']:
        if str(a['image_id']) in need:
            segs.setdefault(str(a['image_id']), []).append(a['segmentation'])
    del data
    if cfg.get('extra_segs_path'):  # E032 union ground truth
        segs.update(json.loads(Path(cfg['extra_segs_path']).read_text(encoding='utf-8')))
    train_ds = DetDataset(cfg['train_items'], segs, size, augment=True, input_mode=INPUT_MODE)
    hard_frac = float(cfg.get('hard_frac', 0.0))
    if hard_frac > 0:  # E070 hard-instance expert
        hard, counts = hard_items(cfg['train_items'], segs, cfg['hard_area'], cfg['hard_thick'], cfg['hard_quantile'])
        sampler = HardMixSampler(len(train_ds), hard, hard_frac, world, rank, SEED)
        if rank == 0:
            log(f'E070 hard-mix sampler: frac {hard_frac}, hard images {len(hard)} / {len(train_ds)}')
            (out / 'e070_hard_items.json').write_text(json.dumps({'hard_frac': hard_frac, 'n': len(train_ds),
                'hard_idx': hard, 'hard_ids': [cfg['train_items'][k][0] for k in hard], 'counts': counts}), encoding='utf-8')
    else:
        sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=SEED)
    nw = cfg['workers']
    train_loader = DataLoader(train_ds, batch_size=1, sampler=sampler, num_workers=nw, pin_memory=use_cuda,
                              persistent_workers=nw > 0, collate_fn=collate_fn, worker_init_fn=seed_worker)
    val_loader = None
    if rank == 0 and cfg['val_items']:  # fold -1 (full data): no validation, last epoch is saved
        val_ds = DetDataset(cfg['val_items'], segs, size, augment=False, input_mode=INPUT_MODE)
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=nw, pin_memory=use_cuda,
                                persistent_workers=nw > 0, collate_fn=collate_fn, worker_init_fn=seed_worker)
    model = build_detector(cfg['pretrained'], size).to(device)
    core = model
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank] if use_cuda else None,
                                                          broadcast_buffers=False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=DET_LR, momentum=DET_MOMENTUM, weight_decay=DET_WD)
    epochs = cfg['epochs']
    milestones = sorted(set([max(1, round(epochs * 0.70)), max(2, round(epochs * 0.90))]))
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=milestones, gamma=0.1)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    status_path = out / 'e006_detector_status.json'
    status = {'rank_world': world, 'milestones': milestones, 'trainable_params': sum(p.numel() for p in params),
              'history': [], 'status': 'training', 'train_images': len(train_ds), 'val_images': len(cfg['val_items'])}
    best, epoch_est = -1.0, None
    flag = torch.zeros(1, device=device if use_cuda else 'cpu')
    for epoch in range(1, epochs + 1):
        if rank == 0:
            flag.zero_()
            if epoch_est is not None and elapsed() + epoch_est > cfg['train_end_by']:
                flag.fill_(1)
                status['stopped_early'] = f'before epoch {epoch}: projected end beyond budget'
                log(status['stopped_early'])
        if distributed:
            dist.broadcast(flag, 0)
        if flag.item():
            break
        sampler.set_epoch(epoch)
        model.train()
        totals = torch.zeros(2, device=device)
        te = time.time()
        steps = 0
        opt.zero_grad(set_to_none=True)
        pending = 0
        for images, targets, _ in train_loader:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) for k, v in t.items()} for t in targets]
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss / accum).backward()
            pending += 1
            if pending == accum:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                pending = 0
            totals[0] += loss.detach().float()
            totals[1] += 1
            steps += 1
            if cfg.get('max_steps') and steps >= cfg['max_steps']:
                break
        if pending:  # last partial accumulation group
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
        if distributed:
            dist.all_reduce(totals)
        train_loss = (totals[0] / totals[1].clamp(min=1)).item()
        train_seconds = time.time() - te
        if rank == 0:
            if not math.isfinite(train_loss):
                flag.fill_(2)
            tv = time.time()
            metrics = source_val(core, val_loader, device, size) if val_loader is not None else {}
            val_seconds = time.time() - tv
            row = {'epoch': epoch, 'train_loss': train_loss, **metrics, 'lr': opt.param_groups[0]['lr'],
                   'train_seconds': train_seconds, 'val_seconds': val_seconds, 'steps_per_rank': steps}
            status['history'].append(row)
            sd = {k: v.detach().cpu() for k, v in core.state_dict().items()}
            if metrics.get('mean_dice', -1) > best:  # source: keep best epoch by fold-0 mean Dice
                best = metrics['mean_dice']
                torch.save({'epoch': epoch, 'model': sd}, out / 'e006_det_best.pt')
                status['best_epoch'] = epoch
            torch.save({'epoch': epoch, 'model': sd}, out / 'e006_det_last.pt')  # last completed epoch
            status['last_epoch'] = epoch
            del sd
            if epoch == 1:
                epoch_est = train_seconds + val_seconds
                projected = cfg['startup_seconds'] + elapsed() + (epochs - 1) * epoch_est + cfg['post_seconds']
                status['first_epoch_check'] = {
                    'epoch_seconds': epoch_est, 'train_seconds': train_seconds, 'val_seconds': val_seconds,
                    'projected_session_seconds': projected, 'limit_seconds': cfg['cap_seconds'] - cfg['reserve_seconds'],
                    'peak_cuda_bytes': torch.cuda.max_memory_allocated(device) if use_cuda else None}
                log(f'detector first-epoch check {status["first_epoch_check"]}')
                if projected > cfg['cap_seconds'] - cfg['reserve_seconds']:
                    flag.fill_(3)
            log(f'detector epoch {row}')
            dump(status_path, status)
        if distributed:
            dist.broadcast(flag, 0)
        if flag.item() == 2:
            status['status'] = 'nonfinite_loss'
            break
        if flag.item() == 3:
            status['status'] = 'aborted_timing_projection'
            break
        sched.step()
        if distributed:
            dist.barrier()
    if rank == 0:
        if status['status'] == 'training':
            status['status'] = 'done'
        status['peak_cuda_bytes'] = torch.cuda.max_memory_allocated(device) if use_cuda else None
        dump(status_path, status)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()


# ------------------------------------------------------------------ CPU prep (refiner crops + GT crops)
def prep_task(task):
    """(path, selected-record segs or None, [segs per annotator record] or None)
    -> (refiner image crops, refiner mask crops, n_empty, gt crops per record or None)."""
    import cv2
    path, sel_segs, entry_segs = task
    image = load_gray(path)
    imgs, msks, empty = [], [], 0
    for seg in sel_segs or []:
        m = seg_to_mask(seg).astype(np.uint8)
        c = trim(0, 0, m.astype(bool))
        if c is None:
            empty += 1
            continue
        x0, y0, x1, y1 = square_bounds(c)
        imgs.append(cv2.resize(image[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_LINEAR))
        msks.append(cv2.resize(m[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_NEAREST))
    gts = None
    if entry_segs is not None:
        gts = [[gt_crop(seg_to_mask(s)) for s in segs] for segs in entry_segs]
    return (np.stack(imgs) if imgs else np.zeros((0, CROP, CROP), np.uint8),
            np.stack(msks) if msks else np.zeros((0, CROP, CROP), np.uint8), empty, gts)


def run_pool(fn, tasks, workers):
    if workers <= 1:
        return [fn(t) for t in tasks]
    import multiprocessing as mp
    with mp.get_context('fork' if sys.platform != 'win32' else 'spawn').Pool(workers) as pool:
        return pool.map(fn, tasks, chunksize=4)


# ------------------------------------------------------------------ refiner stage (cuda:0, as source)
def train_refiner(args, tr_img, tr_msk, va_img, va_msk, device, receipt, rpath):
    import torch
    from torch import nn
    use_cuda = device.type == 'cuda'
    g = torch.Generator().manual_seed(SEED)
    rng = random.Random(SEED)

    def aug_batch(idx):
        xs, ys = [], []
        for i in idx:
            x = torch.from_numpy(tr_img[i].copy()).float() / 255
            y = torch.from_numpy(tr_msk[i].copy()).float().unsqueeze(0)
            k = rng.randrange(4)
            x, y = torch.rot90(x, k, (0, 1)), torch.rot90(y, k, (1, 2))
            if rng.random() < 0.5:
                x, y = torch.flip(x, (1,)), torch.flip(y, (2,))
            if rng.random() < 0.5:
                x = x ** rng.uniform(0.85, 1.20)
            if rng.random() < 0.5:
                x = torch.clamp(x * rng.uniform(0.90, 1.10), 0, 1)
            xs.append(x.unsqueeze(0).repeat(3, 1, 1)); ys.append(y)
        return torch.stack(xs), torch.stack(ys)

    model = build_refiner(args.pretrained).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=REF_LR, weight_decay=REF_WD)
    epochs = args.ref_epochs
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)
    hist, best, info = [], -1.0, {'train_crops': len(tr_img), 'val_crops': len(va_img)}
    for epoch in range(1, epochs + 1):
        te = time.time()
        model.train()
        perm = torch.randperm(len(tr_img), generator=g).tolist()
        ls = bs = ds = n = 0.0
        steps = 0
        for s in range(0, len(perm), REF_BATCH):
            x, y = aug_batch(perm[s:s + REF_BATCH])
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_cuda):
                logits = model(x)
                bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
                p = logits.sigmoid()
                inter = (p * y).sum((1, 2, 3))
                dl = (1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + y.sum((1, 2, 3)) + 1)).mean()
                loss = 0.5 * bce + 0.5 * dl
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            b = len(x)
            ls += loss.item() * b; bs += bce.item() * b; ds += dl.item() * b; n += b
            steps += 1
            if args.max_steps and steps >= args.max_steps:
                break
        if not math.isfinite(ls):
            raise RuntimeError('refiner non-finite loss')
        train_seconds = time.time() - te
        tv = time.time()
        model.eval()
        dsum = vn = 0.0
        with torch.inference_mode():
            for s in range(0, len(va_img), REF_VAL_BATCH):
                x = torch.from_numpy(va_img[s:s + REF_VAL_BATCH]).float().div(255).unsqueeze(1).repeat(1, 3, 1, 1).to(device)
                m = torch.from_numpy(va_msk[s:s + REF_VAL_BATCH]).to(device).bool().unsqueeze(1)
                pr = model(x).sigmoid() >= REFINER_VAL_THRESHOLD
                it = (pr & m).sum((1, 2, 3)).float()
                den = pr.sum((1, 2, 3)) + m.sum((1, 2, 3))
                dsum += ((2 * it + 1e-7) / (den + 1e-7)).sum().item()
                vn += len(x)
        val_dice = dsum / vn if vn else float('nan')
        sched.step()
        row = {'epoch': epoch, 'loss': ls / max(n, 1), 'bce': bs / max(n, 1), 'dice_loss': ds / max(n, 1),
               'val_dice': val_dice, 'train_seconds': train_seconds, 'val_seconds': time.time() - tv}
        hist.append(row)
        sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if val_dice > best:  # source: best epoch by fold-0 crop Dice
            best = val_dice
            torch.save({'epoch': epoch, 'model': sd}, OUT / 'e006_ref_best.pt')
            info['best_epoch'] = epoch
        torch.save({'epoch': epoch, 'model': sd}, OUT / 'e006_ref_last.pt')
        info['last_epoch'] = epoch
        log(f'refiner epoch {row}')
        if epoch == 1:
            est = train_seconds + row['val_seconds']
            projected = args.startup_seconds + elapsed() + (epochs - 1) * est + args.post_refiner_seconds
            info['first_epoch_check'] = {'epoch_seconds': est, 'projected_session_seconds': projected,
                                         'limit_seconds': args.cap_seconds - args.reserve_seconds}
            if projected > args.cap_seconds - args.reserve_seconds:
                info['stopped_early'] = 'refiner first-epoch projection beyond cap - reserve; keeping epoch 1'
                log(info['stopped_early'])
                break
        receipt['refiner'] = {**info, 'history': hist}
        dump(rpath, receipt)
    info['history'] = hist
    return info


# ------------------------------------------------------------------ OOF + post-processing
def load_detector(path, device, size):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = build_detector(False, size)
    m.load_state_dict(ck['model'], strict=True)
    return m.to(device).eval(), int(ck['epoch'])


def load_refiner(path, device):
    import torch
    ck = torch.load(path, map_location='cpu', weights_only=True)
    m = build_refiner(False)
    m.load_state_dict(ck['model'], strict=True)
    return m.to(device).eval(), int(ck['epoch'])


def detector_raw(model, pids, paths, device, size):
    """Raw OOF: per instance (score >= min grid) the mask-probability level map (0..3 = below 0.4 / >=0.4 /
    >=0.5 / >=0.6, compared in fp32 on device) cropped to its nonzero bbox at `size`."""
    import torch
    assert GRID_MASK == (0.40, 0.50, 0.60)
    rec = {'img': [], 'score': [], 'box': [], 'y0': [], 'x0': [], 'h': [], 'w': [], 'flat': []}
    with torch.inference_mode():
        for i, pid in enumerate(pids):
            x = torch.from_numpy(det_input(resize_input(load_gray(paths[pid]), size))).to(device)
            out = model([x])[0]  # fp32, as source inference
            keep = out['scores'] >= min(GRID_SCORE)
            pr = out['masks'][keep, 0]
            lv = (pr >= 0.4).to(torch.uint8) + (pr >= 0.5).to(torch.uint8) + (pr >= 0.6).to(torch.uint8)
            nz = lv > 0
            rows, cols = nz.any(2).cpu().numpy(), nz.any(1).cpu().numpy()
            lv = lv.cpu().numpy()
            for k, (sc, bx) in enumerate(zip(out['scores'][keep].cpu().numpy(), out['boxes'][keep].cpu().numpy())):
                r, c = np.flatnonzero(rows[k]), np.flatnonzero(cols[k])
                y0, y1, x0, x1 = (int(r[0]), int(r[-1]) + 1, int(c[0]), int(c[-1]) + 1) if len(r) else (0, 0, 0, 0)
                crop = lv[k, y0:y1, x0:x1]
                rec['img'].append(i); rec['score'].append(float(sc)); rec['box'].append(bx.tolist())
                rec['y0'].append(y0); rec['x0'].append(x0); rec['h'].append(y1 - y0); rec['w'].append(x1 - x0)
                rec['flat'].append(crop.ravel())
    sizes = [len(f) for f in rec['flat']]
    return {'img': np.asarray(rec['img'], np.int32), 'score': np.asarray(rec['score'], np.float32),
            'box': np.asarray(rec['box'], np.float32).reshape(-1, 4), 'y0': np.asarray(rec['y0'], np.int32),
            'x0': np.asarray(rec['x0'], np.int32), 'h': np.asarray(rec['h'], np.int32), 'w': np.asarray(rec['w'], np.int32),
            'offset': np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64),
            'flat': np.concatenate(rec['flat']).astype(np.uint8) if rec['flat'] else np.zeros(0, np.uint8),
            'size': np.int32(size), 'levels': np.asarray(GRID_MASK, np.float32), 'min_score': np.float32(min(GRID_SCORE))}


def raw_by_image(raw, n_images):
    per = [[] for _ in range(n_images)]
    for k in range(len(raw['img'])):
        a, b = raw['offset'][k], raw['offset'][k + 1]
        per[raw['img'][k]].append((float(raw['score'][k]), int(raw['y0'][k]), int(raw['x0'][k]),
                                   raw['flat'][a:b].reshape(raw['h'][k], raw['w'][k])))
    return per


def detector_masks(insts, score_thr, mask_thr, factor):
    level = 1 + GRID_MASK.index(mask_thr)
    crops = [trim(y0, x0, lv >= level) for s, y0, x0, lv in insts if s >= score_thr and lv.size]
    return [upscale(c, factor) for c in merge_crops(crops)]


class RefinerCache:
    """Refiner probabilities per detector mask (keyed by exact mask); fp32 inference, batch 32 as source."""

    def __init__(self, model, device, image):
        self.model, self.device, self.image, self.cache = model, device, image, {}

    def probs(self, crops):
        import cv2
        import torch
        keys = [(c[0], c[1], c[2].shape, hashlib.blake2b(np.packbits(c[2]).tobytes(), digest_size=16).digest()) for c in crops]
        todo = [i for i, k in enumerate(keys) if k not in self.cache]
        for s in range(0, len(todo), REF_INFER_BATCH):
            idx = todo[s:s + REF_INFER_BATCH]
            bounds = [square_bounds(crops[i]) for i in idx]
            batch = np.stack([cv2.resize(self.image[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_LINEAR)
                              for x0, y0, x1, y1 in bounds])
            with torch.inference_mode():
                t = torch.from_numpy(batch).float().unsqueeze(1).repeat(1, 3, 1, 1).to(self.device) / 255
                pr = self.model(t).sigmoid().cpu().numpy()[:, 0]
            for i, b, p in zip(idx, bounds, pr):
                x0, y0, x1, y1 = b
                self.cache[keys[i]] = (b, cv2.resize(p, (x1 - x0, y1 - y0), interpolation=cv2.INTER_LINEAR))
        return [self.cache[k] for k in keys]


def refined_masks(crops, cache, thr):
    out, fallback = [], 0
    for c, ((x0, y0, x1, y1), p) in zip(crops, cache.probs(crops)):
        t = trim(y0, x0, p >= thr)
        if t is None:
            out.append(c); fallback += 1
        else:
            out.append(t)
    return merge_crops(out), fallback


def score_pipeline(settings, pids, raw_imgs, paths, entries, refiner, device, size, deadline, pq, parity_images=0):
    """settings: list of (score, mask, refiner_thr or None). Exact official PQ semantics (pq_metric)."""
    factor = Fraction(FULL, size)
    mats = {s: [] for s in settings}
    extra = {s: {'pred_instances': 0, 'images_with_preds': 0, 'refiner_fallbacks': 0} for s in settings}
    by_sm = {}
    for s in settings:
        by_sm.setdefault(s[:2], []).append(s[2])
    parity, kept, done, t_first = [], {}, 0, None
    for i, pid in enumerate(pids):
        if elapsed() > deadline:
            break
        ts = time.time()
        image = load_gray(paths[pid]) if refiner is not None and any(r is not None for r in sum(by_sm.values(), [])) else None
        cache = RefinerCache(refiner, device, image) if image is not None else None
        for (sc, mk), refs in by_sm.items():
            det = detector_masks(raw_imgs[i], sc, mk, factor)
            for r in refs:
                s = (sc, mk, r)
                if r is None:
                    preds, fb = det, 0
                else:
                    preds, fb = refined_masks(det, cache, r) if det else ([], 0)
                extra[s]['pred_instances'] += len(preds)
                extra[s]['images_with_preds'] += bool(preds)
                extra[s]['refiner_fallbacks'] += fb
                if s == settings[0]:
                    kept[pid] = preds
                for gts in entries[pid]:
                    m = overlap(gts, preds)
                    mats[s].append(m)
                    if i < parity_images and s == settings[0] and len(gts) <= 25 and len(preds) <= 25:
                        gd = np.stack([dense((g[0], g[1], g[2])) if g[3] else np.zeros((FULL, FULL), np.uint8) for g in gts]) \
                            if gts else np.zeros((0, FULL, FULL), np.uint8)
                        pd_ = np.stack([dense(p) for p in preds]) if preds else np.zeros((0, FULL, FULL), np.uint8)
                        ref = pq.overlap_matrix(gd, pd_)
                        parity.append({'image': pid, 'equal': bool(ref.dtype == m.dtype and np.array_equal(ref, m)),
                                       'shape': list(m.shape)})
        done += 1
        if t_first is None:
            t_first = time.time() - ts
    results = {}
    for s in settings:
        res = pq.score_matrices(mats[s])
        res.update(extra[s])
        res['images_scored'] = done
        results[s] = res
    return results, {'images_scored': done, 'images_total': len(pids), 'complete': done == len(pids),
                     'first_image_seconds': t_first, 'parity': parity}, kept


def setting_name(s):
    return {'score_threshold': s[0], 'mask_threshold': s[1], 'refiner_threshold': s[2]}


def validator_roundtrip(kept, ids, sub, vs, n_images=3):
    """Encode the first images' final instances as competition RLE rows, run validate_submission, decode back."""
    from pycocotools import mask as cm
    ids3 = [p for p in ids if kept.get(p)][:n_images]
    rows, masks = [], {}
    for pid in ids3:
        for k, c in enumerate(kept[pid], 1):
            m = np.asfortranarray(dense(c))
            rows.append((f'{pid}_{k}', cm.encode(m)['counts'].decode('ascii')))
            masks[f'{pid}_{k}'] = m
    with open(sub, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filament_id', 'segmentation_rle'])
        w.writerows(rows)
    rt = {'images': ids3, 'rows': len(rows), 'passed': None}
    if rows:
        try:
            rt['validator'] = vs.validate_submission(sub, allowed_ids=ids3, reject_empty_masks=True, reject_all_empty=True)
        except ValueError as exc:  # recorded, never allowed to discard the grid that follows
            rt.update(validator={'valid': False, 'error': str(exc)}, passed=False)
            return rt
        csv.field_size_limit(64 * 1024 * 1024)
        with open(sub, newline='', encoding='utf-8') as f:
            rt['decode_mismatches'] = sum(
                not np.array_equal(cm.decode({'size': [FULL, FULL], 'counts': r['segmentation_rle'].encode('ascii')}),
                                   masks[r['filament_id']]) for r in csv.DictReader(f))
        rt['passed'] = bool(rt['validator'].get('valid') and rt['decode_mismatches'] == 0)
    return rt


# ------------------------------------------------------------------ main (orchestrator)
OUT = None


def weight_provenance(pretrained):
    import torch
    from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights
    from torchvision.models import ResNet18_Weights
    info = {'pretrained': pretrained}
    if not pretrained:
        return info
    t = time.time()
    for name, w in (('detector', MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT), ('refiner_encoder', ResNet18_Weights.DEFAULT)):
        w.get_state_dict(progress=False, check_hash=True)  # torchvision's own URL + hash check (internet)
        f = Path(torch.hub.get_dir()) / 'checkpoints' / os.path.basename(w.url)
        info[name] = {'enum': str(w), 'url': w.url, 'file': f.name, 'sha256': sha_file(f) if f.exists() else None,
                      'bytes': f.stat().st_size if f.exists() else None,
                      'license': 'torchvision code BSD-3-Clause; weights per torchvision model docs'}
    info['detector']['training_data'] = 'COCO 2017 train (torchvision recipe); no solar data'
    info['refiner_encoder']['training_data'] = 'ImageNet-1k (torchvision IMAGENET1K_V1)'
    info['download_seconds'] = time.time() - t
    return info


def main():
    global OUT, LOG_PATH, SEED
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    ap.add_argument('--folds-csv', default=None)
    ap.add_argument('--fold', type=int, default=0)
    ap.add_argument('--image-size', type=int, default=IMAGE_SIZE, help='source 1024; smaller only for CPU smoke')
    ap.add_argument('--seed', type=int, default=SEED, help='source 2026; B0 seed rerun uses another')
    ap.add_argument('--train-records', default='selected', choices=['selected', 'random', 'union'],
                    help='detector training labels: source = record with most filaments; random = any record per sample (B1)')
    ap.add_argument('--cudnn-benchmark', default='off', choices=['on', 'off'], help='on = source behaviour (slow with Mask R-CNN)')
    ap.add_argument('--input-mode', default='gray', choices=['gray', 'phys'], help='detector input channels (E034 phys)')
    ap.add_argument('--hard-frac', type=float, default=0.0, help='E070: share of each epoch drawn from hard images (0 = uniform)')
    ap.add_argument('--accum', type=int, default=1, help='gradient accumulation; 2 on one GPU == source DDP 2x batch 1')
    ap.add_argument('--det-epochs', type=int, default=DET_EPOCHS)
    ap.add_argument('--ref-epochs', type=int, default=REF_EPOCHS)
    ap.add_argument('--max-steps', type=int, default=0, help='smoke only: cap steps per epoch')
    ap.add_argument('--max-val-images', type=int, default=0, help='smoke only')
    ap.add_argument('--pretrained', default='yes', choices=['yes', 'no'])
    ap.add_argument('--device', default='auto')
    ap.add_argument('--nproc', type=int, default=-1, help='DDP ranks (-1 = all GPUs; 0 = in-process, no DDP)')
    ap.add_argument('--det-workers', type=int, default=2, help='source: 2 per rank')
    ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1), help='CPU prep pool')
    ap.add_argument('--cap-seconds', type=float, default=10800, help='server session cap (3 h)')
    ap.add_argument('--reserve-seconds', type=float, default=900)
    ap.add_argument('--startup-seconds', type=float, default=180, help='session time before script start')
    ap.add_argument('--post-seconds', type=float, default=1800, help='estimate after detector: crops+refiner+OOF+default PQ')
    ap.add_argument('--post-refiner-seconds', type=float, default=900, help='estimate after refiner: OOF+default PQ')
    ap.add_argument('--require-gpu', action='store_true')
    ap.add_argument('--detector-worker', default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    SEED = args.seed
    global INPUT_MODE
    INPUT_MODE = args.input_mode
    if args.detector_worker:
        return detector_worker(args.detector_worker)
    args.pretrained = args.pretrained == 'yes'
    if EMBEDDED and args.device == 'auto':
        args.require_gpu = True  # Kaggle build: never silently train on CPU
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    LOG_PATH = OUT / 'train.log'
    receipt = {'experiment': 'E006', 'mode': 'validation', 'fold': args.fold, 'status': 'starting',
               'source_notebook': SOURCE_REF, 'source_snapshot_sha256': SOURCE_SNAPSHOT_SHA256,
               'source_sha256': sha_file(__file__), 'deviations': DEVIATIONS,
               'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               'hyperparameters': {'image_size': args.image_size, 'det_epochs': args.det_epochs, 'det_lr': DET_LR,
                                   'ref_epochs': args.ref_epochs, 'ref_lr': REF_LR, 'defaults': setting_name(DEFAULT),
                                   'grid': {'score': GRID_SCORE, 'mask': GRID_MASK, 'refiner': GRID_REF}}}
    receipt['embedded_sha256'] = setup_lib(OUT)
    rpath = OUT / 'e006_receipt.json'
    dump(rpath, receipt)
    hard = args.cap_seconds - args.startup_seconds - args.reserve_seconds  # watchdog: 15 min session reserve
    child = {'proc': None}

    def watchdog():
        try:
            p = child['proc']
            if p is not None and p.poll() is None:
                try:
                    os.killpg(p.pid, 9)
                except Exception:
                    p.kill()
            last = json.loads(rpath.read_text(encoding='utf-8'))
            last.update(status='watchdog_timeout', terminal=True, watchdog_seconds=hard)
            dump(rpath, last)
        finally:
            os._exit(124)
    timer = threading.Timer(max(1.0, hard - elapsed()), watchdog)
    timer.daemon = True
    timer.start()
    try:
        _main(args, receipt, rpath, child, hard)
    except Exception as exc:
        import traceback
        receipt.update(status='exception', exception=repr(exc), traceback=traceback.format_exc()[-4000:], terminal=True)
        dump(rpath, receipt)
        log(f'EXCEPTION {exc!r}')
        raise
    finally:
        timer.cancel()


def _main(args, receipt, rpath, child, hard):
    import torch
    import pq_metric as pq
    import validate_submission as vs
    receipt['versions'] = versions()
    seed_everything(SEED)
    log(f'versions {receipt["versions"]}')

    # ---- data + folds (C1)
    folds, folds_sha = read_folds(args)
    receipt['folds_csv_sha256'] = folds_sha
    ann_path, images = discover(args.input)
    rec_phys, rec_anns = load_annotations(ann_path, images)
    receipt['annotation_sha256'] = sha_file(ann_path)
    if set(rec_phys) != set(folds):
        raise ValueError(f'folds/annotation record mismatch: {len(set(rec_phys) ^ set(folds))}')
    phys_fold, phys_recs = {}, {}
    for rid, pid in sorted(rec_phys.items()):
        fp, k = folds[rid]
        if fp != pid or phys_fold.setdefault(pid, k) != k:
            raise ValueError(f'fold inconsistency at {rid}')
        if rid in rec_anns:
            phys_recs.setdefault(pid, []).append(rid)
    pids = sorted(phys_recs)
    full = args.fold < 0  # E031 full-data model: train on every image, no validation / OOF / PQ
    if full and args.fold != -1:
        raise ValueError('bad fold')
    val_ids = [] if full else [p for p in pids if phys_fold[p] == args.fold]
    tr_ids = list(pids) if full else [p for p in pids if phys_fold[p] != args.fold]
    if args.max_val_images:
        val_ids = val_ids[:args.max_val_images]
    if full:
        if not tr_ids:
            raise ValueError('bad split')
        receipt['mode'] = 'full_data'
    elif not val_ids or not tr_ids or set(phys_fold[p] for p in tr_ids) & {args.fold}:
        raise ValueError('bad split')
    selected = {p: select_record(phys_recs[p], rec_anns) for p in pids}
    paths = {p: str(images[p][1]) for p in pids}
    receipt['data'] = {'train_images': len(tr_ids), 'train_folds': sorted({phys_fold[p] for p in tr_ids}),
                       'val_images': len(val_ids), 'val_records': sum(len(phys_recs[p]) for p in val_ids),
                       'val_instances': sum(len(rec_anns[r]) for p in val_ids for r in phys_recs[p]),
                       'train_selected_instances': sum(len(rec_anns[selected[p]]) for p in tr_ids),
                       'train_records_mode': args.train_records,
                       'train_records_total': sum(len(phys_recs[p]) for p in tr_ids)}
    log(f'data {receipt["data"]}')
    dump(OUT / 'e006_split.json', {'train': {p: selected[p] for p in tr_ids}, 'val': {p: selected[p] for p in val_ids},
                                   'folds_csv_sha256': folds_sha})

    union_ids = {}
    if args.train_records == 'union':
        tu = time.time()
        order = {p: sorted(phys_recs[p], key=lambda r: (len(rec_anns[r]), sum(float(a['area']) for a in rec_anns[r]), str(r)),
                           reverse=True) for p in tr_ids}
        assert all(order[p][0] == selected[p] for p in tr_ids)
        res = run_pool(union_record, [(p, [[a['segmentation'] for a in rec_anns[r]] for r in order[p]]) for p in tr_ids],
                       args.workers)
        audits = {pid: au for pid, _, au in res}
        summary = {'images': len(res), 'multi_record_images': sum(a['records'] > 1 for a in audits.values()),
                   'selected_instances': sum(a['selected_instances'] for a in audits.values()),
                   'union_instances': sum(a['out_instances'] for a in audits.values()),
                   'single_member_clusters': sum(a['single_member'] for a in audits.values()),
                   'clusters': sum(a['clusters'] for a in audits.values()),
                   'removed_empty': sum(a['removed_empty'] for a in audits.values()),
                   'max_disputed_share': max(a['disputed_share'] for a in audits.values()),
                   'images_disputed_over_1pct': sum(a['disputed_share'] > 0.01 for a in audits.values()),
                   'seconds': time.time() - tu}
        receipt['union_audit'] = summary
        dump(OUT / 'e006_union_audit.json', {'summary': summary, 'per_image': audits})
        log(f'union audit {summary}')
        if summary['images_disputed_over_1pct']:
            raise RuntimeError(f'union audit: {summary["images_disputed_over_1pct"]} images with disputed pixels > 1%')
        dump(OUT / 'e006_union_segs.json', {f'U{pid}': rles for pid, rles, _ in res})
        union_ids = {pid: f'U{pid}' for pid, _, _ in res}
    receipt['input_mode'] = args.input_mode
    receipt['hard_frac'] = args.hard_frac
    receipt['cudnn_benchmark'] = args.cudnn_benchmark

    # ---- weights (C3: internet used only here)
    receipt['weights'] = weight_provenance(args.pretrained)
    dump(rpath, receipt)
    log(f'weights {receipt["weights"]}')

    # ---- detector stage (DDP subprocess; CUDA is not initialised in this process yet)
    n_gpu = torch.cuda.device_count() if args.device in ('auto', 'cuda') else 0
    if args.require_gpu and n_gpu == 0:
        raise RuntimeError('GPU required but not available')
    dev_type = 'cuda' if n_gpu else 'cpu'
    nproc = (n_gpu or 1) if args.nproc < 0 else args.nproc
    train_end_by = args.cap_seconds - args.reserve_seconds - args.startup_seconds - args.post_seconds
    cfg = {'t0_wall': T0_WALL, 'out': str(OUT), 'device': dev_type, 'image_size': args.image_size,
           'epochs': args.det_epochs, 'pretrained': args.pretrained, 'workers': args.det_workers,
           'ann_path': str(ann_path), 'max_steps': args.max_steps, 'seed': SEED, 'accum': args.accum,
           'train_items': [(p, paths[p], list(phys_recs[p]) if args.train_records == 'random' else union_ids.get(p, selected[p]))
                           for p in tr_ids],
           'input_mode': args.input_mode, 'cudnn_benchmark': args.cudnn_benchmark,
           'hard_frac': args.hard_frac, 'hard_area': 2000, 'hard_thick': 8, 'hard_quantile': 0.75,
           'extra_segs_path': str(OUT / 'e006_union_segs.json') if union_ids else None,
           'val_items': [(p, paths[p], selected[p]) for p in val_ids],
           'cap_seconds': args.cap_seconds, 'reserve_seconds': args.reserve_seconds,
           'startup_seconds': args.startup_seconds, 'post_seconds': args.post_seconds, 'train_end_by': train_end_by}
    cfg_path = OUT / 'e006_detector_config.json'
    dump(cfg_path, cfg)
    receipt['device'] = {'type': dev_type, 'cuda_device_count': n_gpu, 'ddp_ranks': nproc}  # no CUDA init here (fork pool later)
    receipt.update(status='training_detector', budget={'cap_seconds': args.cap_seconds, 'reserve_seconds': args.reserve_seconds,
                                                       'startup_seconds_assumed': args.startup_seconds,
                                                       'detector_must_end_by_script_seconds': train_end_by,
                                                       'watchdog_script_seconds': hard})
    receipt['timing'] = {}
    dump(rpath, receipt)
    td = time.time()
    me = str(Path(__file__).resolve())
    if nproc >= 1:
        cmd = [sys.executable, '-m', 'torch.distributed.run', '--standalone', f'--nproc_per_node={nproc}']
        if os.name != 'posix':  # local Windows smoke only: CPU wheels lack libuv
            cmd.append('--rdzv-conf=use_libuv=0')
        cmd += [me, '--detector-worker', str(cfg_path)]
    else:
        cmd = None
    if cmd:
        with open(OUT / 'detector_ddp.log', 'a', encoding='utf-8') as lf:
            kw = {'start_new_session': True} if os.name == 'posix' else {}
            child['proc'] = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, **kw)
            rc = child['proc'].wait()
        child['proc'] = None
        receipt['detector_returncode'] = rc
    else:
        detector_worker(str(cfg_path))
        rc = 0
    receipt['timing']['detector_stage_seconds'] = time.time() - td
    st_path = OUT / 'e006_detector_status.json'
    det = json.loads(st_path.read_text(encoding='utf-8')) if st_path.exists() else {}
    receipt['detector'] = det
    dump(rpath, receipt)
    if rc != 0 or det.get('status') != 'done' or not (OUT / 'e006_det_last.pt').exists():
        tail = (OUT / 'detector_ddp.log').read_text(encoding='utf-8', errors='replace')[-3000:] if (OUT / 'detector_ddp.log').exists() else ''
        receipt.update(status=f'detector_{det.get("status", "failed")}', terminal=True, detector_log_tail=tail)
        dump(rpath, receipt)
        log(f'STOP: detector stage rc={rc} status={det.get("status")}')
        return
    log(f'detector done: last epoch {det.get("last_epoch")} best epoch {det.get("best_epoch")}')

    # ---- CPU prep: refiner crops (selected records) + GT crops (all annotator records of fold 0)
    tp = time.time()
    tasks = [(paths[p], [a['segmentation'] for a in rec_anns[selected[p]]], None) for p in tr_ids] + \
            [(paths[p], [a['segmentation'] for a in rec_anns[selected[p]]],
              [[a['segmentation'] for a in rec_anns[r]] for r in phys_recs[p]]) for p in val_ids]
    prepped = run_pool(prep_task, tasks, args.workers)
    ntr = len(tr_ids)
    tr_img = np.concatenate([x[0] for x in prepped[:ntr]]); tr_msk = np.concatenate([x[1] for x in prepped[:ntr]])
    if val_ids:
        va_img = np.concatenate([x[0] for x in prepped[ntr:]]); va_msk = np.concatenate([x[1] for x in prepped[ntr:]])
    else:  # full data: refiner has no validation crops (val Dice nan, last epoch kept)
        va_img = np.zeros((0, CROP, CROP), np.uint8); va_msk = np.zeros((0, CROP, CROP), np.uint8)
    entries = {p: prepped[ntr + j][3] for j, p in enumerate(val_ids)}
    receipt['timing']['crop_cache_seconds'] = time.time() - tp
    receipt['refiner_data'] = {'train_crops': len(tr_img), 'val_crops': len(va_img),
                               'empty_annotation_masks_skipped': sum(x[2] for x in prepped),
                               'val_entries': sum(len(v) for v in entries.values())}
    del prepped
    log(f'crop cache {receipt["refiner_data"]} in {receipt["timing"]["crop_cache_seconds"]:.1f}s')

    device = torch.device('cuda:0' if dev_type == 'cuda' else 'cpu')  # first CUDA use in this process (after fork pool)
    receipt['device']['cuda_devices'] = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    torch.backends.cudnn.benchmark = args.cudnn_benchmark == 'on'
    tr = time.time()
    receipt['status'] = 'training_refiner'
    dump(rpath, receipt)
    receipt['refiner'] = train_refiner(args, tr_img, tr_msk, va_img, va_msk, device, receipt, rpath)
    receipt['timing']['refiner_seconds'] = time.time() - tr
    del tr_img, tr_msk, va_img, va_msk

    # ---- checkpoints recorded; drop duplicate best files
    ck = {}
    for kind in ('det', 'ref'):
        last, best = OUT / f'e006_{kind}_last.pt', OUT / f'e006_{kind}_best.pt'
        le = (receipt['detector'] if kind == 'det' else receipt['refiner']).get('last_epoch')
        be = (receipt['detector'] if kind == 'det' else receipt['refiner']).get('best_epoch')
        if best.exists() and be == le:
            best.unlink()
        ck[kind] = {'last': {'path': last.name, 'epoch': le, 'sha256': sha_file(last)},
                    'best': ({'path': best.name, 'epoch': be, 'sha256': sha_file(best)} if best.exists()
                             else {'same_as_last': True, 'epoch': be})}
    receipt['checkpoints'] = ck
    receipt['status'] = 'checkpoints_saved'
    dump(rpath, receipt)
    if full:  # no OOF / PQ / round-trip; test fusion reads the input size from the receipt
        receipt['image_size'] = args.image_size
        receipt.update(status='complete', terminal=True)
        receipt['timing']['script_seconds'] = elapsed()
        dump(rpath, receipt)
        log('SUMMARY ' + json.dumps({'status': 'complete', 'mode': 'full_data', 'det_epochs': det.get('last_epoch'),
                                     'ref_epochs': receipt['refiner'].get('last_epoch'), 'checkpoints': ck,
                                     'timing': receipt['timing']}, default=str))
        return

    # ---- raw detector OOF (checkpoints reloaded from disk into weight-free models = reload check)
    to = time.time()
    raws = {}
    for tag in ('last', 'best'):
        if tag == 'best' and ck['det']['best'].get('same_as_last'):
            continue
        model, ep = load_detector(OUT / ck['det'][tag]['path'], device, args.image_size)
        raw = detector_raw(model, val_ids, paths, device, args.image_size)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        p = OUT / f'e006_fold{args.fold}_oof_det_{tag}.npz'
        np.savez(p, **raw)
        raws[tag] = raw
        receipt.setdefault('oof', {})[tag] = {'path': p.name, 'epoch': ep, 'instances': int(len(raw['img'])),
                                              'sha256': sha_file(p)}
    dump(OUT / f'e006_fold{args.fold}_oof_ids.json', val_ids)
    receipt['timing']['oof_inference_seconds'] = time.time() - to
    receipt['status'] = 'oof_saved'
    dump(rpath, receipt)
    log(f'raw OOF saved {receipt["oof"]}')

    # ---- fixed-default PQ (source defaults), primary = last/last; source-faithful = best/best
    grid_deadline = hard - 120
    ref_last, _ = load_refiner(OUT / ck['ref']['last']['path'], device)
    raw_last = raw_by_image(raws['last'], len(val_ids))
    tq = time.time()
    res, info, kept = score_pipeline([DEFAULT, DEFAULT_DET_ONLY], val_ids, raw_last, paths, entries, ref_last, device,
                                     args.image_size, grid_deadline, pq, parity_images=2)
    gt_mean = float(np.mean([np.mean([len(g) for g in entries[p]]) for p in val_ids]))
    receipt['metrics'] = {
        'pq_fixed_default_primary': {'checkpoints': 'last detector + last refiner', **setting_name(DEFAULT), **res[DEFAULT],
                                     'pred_per_image_mean': res[DEFAULT]['pred_instances'] / max(1, info['images_scored'])},
        'pq_fixed_default_detector_only': {**setting_name(DEFAULT_DET_ONLY), **res[DEFAULT_DET_ONLY]},
        'pq_default_run': info, 'gt_mean_instances_per_entry_image': gt_mean}
    from pycocotools import mask as cm
    dump(OUT / f'e006_fold{args.fold}_oof_instances_default.json',
         {pid: [cm.encode(np.asfortranarray(dense(c)))['counts'].decode('ascii') for c in cs] for pid, cs in kept.items()})
    if not ck['det']['best'].get('same_as_last') or not ck['ref']['best'].get('same_as_last'):
        raw_b = raw_by_image(raws.get('best', raws['last']), len(val_ids))
        ref_b = ref_last if ck['ref']['best'].get('same_as_last') else load_refiner(OUT / ck['ref']['best']['path'], device)[0]
        rb, ib, _ = score_pipeline([DEFAULT, DEFAULT_DET_ONLY], val_ids, raw_b, paths, entries, ref_b, device,
                                   args.image_size, grid_deadline, pq)
        receipt['metrics']['pq_fixed_default_source_faithful'] = {
            'checkpoints': f'detector epoch {ck["det"]["best"]["epoch"]} + refiner epoch {ck["ref"]["best"]["epoch"]} '
                           '(best on fold 0: selection-biased)', **setting_name(DEFAULT), **rb[DEFAULT],
            'detector_only': rb[DEFAULT_DET_ONLY], 'complete': ib['complete']}
    else:
        receipt['metrics']['pq_fixed_default_source_faithful'] = 'identical to primary (best epoch == last epoch)'
    receipt['timing']['pq_default_seconds'] = time.time() - tq
    d = res[DEFAULT]
    dump(rpath, receipt)
    log(f'fixed-default PQ {d["pq"]:.4f} (det-only {res[DEFAULT_DET_ONLY]["pq"]:.4f}) tp {d["tp"]} fp {d["fp"]} fn {d["fn"]} '
        f'images {d["images_scored"]}/{len(val_ids)} parity {info["parity"]}')

    # ---- validator round-trip on 3 fold-0 images (RLE encode -> validate -> decode == mask)
    rt = validator_roundtrip(kept, val_ids, OUT / 'e006_roundtrip_oof.csv', vs)
    receipt['roundtrip'] = rt
    dump(rpath, receipt)
    log(f'round-trip {rt.get("passed")} rows {rt["rows"]}')

    # ---- time-bounded PQ grid on the primary pipeline (tuned = selection-biased)
    tg = time.time()
    settings = [(s, m, r) for s in GRID_SCORE for m in GRID_MASK for r in GRID_REF]
    settings.sort(key=lambda s: s != DEFAULT)
    grid, ginfo, _ = score_pipeline(settings, val_ids, raw_last, paths, entries, ref_last, device, args.image_size,
                                    grid_deadline, pq)
    rows = [{**setting_name(s), **{k: v[k] for k in ('pq', 'tp', 'fp', 'fn', 'tp_iou_sum', 'pred_instances',
                                                     'refiner_fallbacks', 'images_scored')}} for s, v in grid.items()]
    rows.sort(key=lambda x: -x['pq'])
    with open(OUT / 'e006_pq_grid.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    receipt['pq_grid'] = {'complete': ginfo['complete'], 'images_scored': ginfo['images_scored'],
                          'first_image_seconds': ginfo['first_image_seconds'], 'seconds': time.time() - tg,
                          'top10': rows[:10], 'file': 'e006_pq_grid.csv',
                          'note': 'tuned PQ is selected on the same fold-0 OOF it is reported on: selection-biased'}
    if ginfo['complete']:
        receipt['metrics']['pq_tuned_selection_biased'] = rows[0]
    receipt.update(status='complete', terminal=True)
    receipt['timing']['script_seconds'] = elapsed()
    dump(rpath, receipt)
    summary = {'status': 'complete', 'det_epochs': det.get('last_epoch'), 'stopped_early': det.get('stopped_early'),
               'first_epoch_check': det.get('first_epoch_check'), 'timing': receipt['timing'],
               'pq_fixed_default_primary': d['pq'], 'pq_detector_only_default': res[DEFAULT_DET_ONLY]['pq'],
               'pq_tuned_selection_biased': rows[0]['pq'] if ginfo['complete'] else None, 'roundtrip': rt.get('passed')}
    log('SUMMARY ' + json.dumps(summary, default=str))


if __name__ == '__main__':
    main()
