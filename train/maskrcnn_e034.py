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
EMBEDDED = {'pq_metric.py': '"""NumPy reproduction of official self-evaluation notebook cells 3-4.\n\nSource: snapshots/official-self-evaluation/self-evaluation-notebook.ipynb.\nThis establishes local source semantics only, not online-service equivalence.\nRepeated annotator entries reuse physical-image predictions. All qualifying\npairs count, with no one-to-one assignment. Extra-only physical images are\nignored by the source entry construction; zero-GT entries cannot originate\nfrom an annotation-row table. Do not exploit either limitation.\n"""\nimport numpy as np\n\n\ndef overlap_matrix(gt, pred):\n    gt, pred = np.asarray(gt), np.asarray(pred)\n    if gt.ndim != 3 or pred.ndim != 3 or gt.shape[1:] != pred.shape[1:]:\n        raise ValueError(\'Expected matching (instances,height,width) arrays\')\n    if not np.isin(gt, [0, 1]).all() or not np.isin(pred, [0, 1]).all():\n        raise ValueError(\'Masks must be binary\')\n    # Official arrays are float32; preserve that arithmetic rather than uint8\n    # matmul (which silently overflows) or a differently rounded float64 path.\n    g = gt.astype(np.float32).reshape(len(gt), -1) if len(gt) else np.empty((0, np.prod(gt.shape[1:])), np.float32)\n    p = pred.astype(np.float32).reshape(len(pred), -1) if len(pred) else np.empty((0, np.prod(pred.shape[1:])), np.float32)\n    intersection = g @ p.T\n    union = g.sum(axis=1)[:, None] + p.sum(axis=1)[None, :] - intersection\n    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)\n\n\ndef score_matrices(matrices):\n    """Globally aggregate overlap matrices; preserve explicit empty-entry branch."""\n    tp, fp, fn, tp_values = 0, 0, 0, []\n    for matrix in matrices:\n        matrix = np.asarray(matrix)\n        if matrix.ndim != 2 or not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():\n            raise ValueError(\'Invalid IoU matrix\')\n        n_gt, n_pred = matrix.shape\n        if n_gt == 0:\n            fp += n_pred\n            continue\n        if n_pred == 0:\n            fn += n_gt\n            continue\n        hit = matrix > .5\n        values = matrix[hit].tolist()\n        tp += len(values)\n        tp_values.extend(values)\n        fp += int((hit.sum(axis=0) == 0).sum())\n        fn += int((hit.sum(axis=1) == 0).sum())\n    denominator = tp + .5 * fp + .5 * fn\n    iou_sum = sum(tp_values)\n    return {\'pq\': iou_sum / denominator if denominator else 0.0, \'tp\': tp, \'fp\': fp, \'fn\': fn, \'tp_iou_sum\': iou_sum}\n\n\ndef evaluate_entries(entries, predictions):\n    """entries: iterable (annotator-image ID, physical ID, GT stack).\n\n    Caller supplies one entry per annotator-image represented in annotation\n    rows, matching official get_overlap_df construction. Repeated physical IDs\n    intentionally reuse predictions. Missing predictions become empty stacks.\n    """\n    matrices, seen, used = [], set(), set()\n    for entry_id, physical_id, gt in entries:\n        if entry_id in seen:\n            raise ValueError(\'Duplicate annotator-image entry\')\n        seen.add(entry_id)\n        gt = np.asarray(gt)\n        if gt.ndim != 3:\n            raise ValueError(\'Invalid GT shape\')\n        pred = predictions.get(physical_id, np.zeros((0, *gt.shape[1:]), dtype=np.uint8))\n        matrices.append(overlap_matrix(gt, pred))\n        used.add(physical_id)\n    result = score_matrices(matrices)\n    result.update(entries=len(seen), ignored_extra_prediction_images=sorted(set(predictions) - used),\n                  scope=\'official local source cells 3-4; online scorer parity unknown\')\n    return result\n', 'gpu_smoke_audit.py': '"""Original offline Kaggle audit + tiny training smoke test; never submits or infers test."""\nimport argparse\nimport csv\nimport hashlib\nimport importlib.metadata\nimport json\nimport math\nimport os\nfrom pathlib import Path\nimport re\nimport threading\nimport time\nimport traceback\n\nSTART = time.monotonic()\n\n\ndef dump(path, value):\n    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")\n\n\ndef terminal_receipt(out, status, details):\n    path = out / "audit.json"\n    try:\n        audit = json.loads(path.read_text(encoding="utf-8"))\n    except (OSError, ValueError):\n        audit = {}\n    audit.update(status=status, baseline_ready=False, full_audit_complete=False,\n                 total_seconds=time.monotonic() - START, terminal_details=details)\n    dump(path, audit)\n\n\ndef write_csv(path, rows):\n    if not rows:\n        raise ValueError(f"No rows for {path}")\n    with path.open("w", newline="", encoding="utf-8") as stream:\n        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))\n        writer.writeheader()\n        writer.writerows(rows)\n\n\ndef parse_name(path):\n    match = re.fullmatch(r"(\\d{8})(\\d{6})([A-Za-z]{2})", path.stem)\n    if not match:\n        raise ValueError(f"Unexpected physical image name: {path.name}")\n    from datetime import datetime\n    datetime.strptime(match[1] + match[2], "%Y%m%d%H%M%S")\n    return match[1], match[3]\n\n\ndef main(args):\n    # Import heavy dependencies only after the hard wall-clock watchdog is active.\n    import numpy as np\n    import PIL\n    from PIL import Image\n    import torch\n    from torch import nn\n    from pycocotools import mask as coco_mask\n\n    out = args.output\n    effective_config = {"steps": args.steps, "wall_seconds": args.timeout,\n                        "input": str(args.input), "output": str(args.output),\n                        "requested_accelerator": "NvidiaTeslaT4", "server_timeout_expected_seconds": 450,\n                        "maximum_allocated_devices": 2, "maximum_aggregate_gpu_hours": 0.25,\n                        "architecture": "original Conv1-8-8-1", "seed": 2026,\n                        "image_resize": [256, 256], "maximum_batch": 4}\n    audit = {"purpose": "environment and partial data audit only; no baseline CV", "status": "running",\n             "baseline_ready": False, "full_audit_complete": False,\n             "stages": {"structural_audit": "running", "mask_decoding": "pending", "training": "pending", "checkpoint": "pending"},\n             "outstanding_checks": ["full mask decoding and foreground distributions", "category/bbox/area/license integrity", "near-duplicate audit", "final balanced folds", "official metric parity", "held-out inference and checkpoint reload/export"],\n             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\n             "effective_config": effective_config,\n             "config_sha256": hashlib.sha256(json.dumps(effective_config, sort_keys=True).encode()).hexdigest(),\n             "limits": {"steps": args.steps, "wall_seconds": args.timeout},\n             "versions": {"torch": torch.__version__, "numpy": np.__version__, "Pillow": PIL.__version__, "pycocotools": importlib.metadata.version("pycocotools")}}\n    dump(out / "audit.json", audit)\n    if not torch.cuda.is_available():\n        raise RuntimeError("CUDA unavailable: refusing CPU/local training fallback")\n    audit["cuda"] = {"version": torch.version.cuda, "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}\n    dump(out / "audit.json", audit)\n    if torch.cuda.device_count() not in (1, 2):\n        raise RuntimeError("This T4 diagnostic requires one or two allocated CUDA devices")\n    candidates = []\n    for path in args.input.rglob("*.json"):\n        if "train" in path.name.lower() and "annotation" in path.name.lower():\n            candidates.append(path)\n    if len(candidates) != 1:\n        raise ValueError(f"Expected exactly one train annotation JSON; found {candidates}")\n    annotation_path = candidates[0]\n    data = json.loads(annotation_path.read_text(encoding="utf-8"))\n    if not isinstance(data, dict) or not isinstance(data.get("images"), list) or not isinstance(data.get("annotations"), list):\n        raise ValueError("Annotation JSON must contain COCO images and annotations lists")\n    if not data["images"] or not data["annotations"]:\n        raise ValueError("Empty training images or annotations")\n    files = {}\n    split_for = {}\n    for path in args.input.rglob("*"):\n        if path.suffix.lower() not in {".jpg", ".jpeg"}:\n            continue\n        dirs = {p.lower() for p in path.parts}\n        split = "train" if "train_images" in dirs else "test" if "test_images" in dirs else None\n        if split is None:\n            continue\n        if path.name in files or path.stem in split_for:\n            raise ValueError(f"Ambiguous duplicate mounted image name: {path}")\n        files[path.name] = path\n        split_for[path.stem] = split\n    if set(split_for.values()) != {"train", "test"}:\n        raise ValueError("Expected mounted train_images and test_images JPEG directories")\n    physical = {}\n    for name, path in sorted(files.items()):\n        day, station = parse_name(path)\n        with Image.open(path) as im:\n            if im.size != (2048, 2048):\n                raise ValueError(f"Invalid image dimensions: {path}: {im.size}")\n            if im.mode not in ("L", "RGB"):\n                raise ValueError(f"Unexpected JPEG mode {im.mode}: {path}")\n            if im.mode == "RGB":\n                rgb = np.asarray(im)\n                if not (np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])):\n                    raise ValueError(f"Non-grayscale RGB image: {path}")\n            pixels = np.asarray(im.convert("L"))\n            digest = hashlib.sha256(pixels.tobytes()).hexdigest()\n        physical[path.stem] = {"physical_id": path.stem, "split": split_for[path.stem], "day": day,\n                               "station": station, "pixel_sha256": digest, "path": str(path)}\n    records = {}\n    counts = {}\n    annotations = {}\n    for row in data["images"]:\n        key = str(row["id"])\n        name = Path(row["file_name"]).name\n        if key in records or name not in files or split_for.get(Path(name).stem) != "train":\n            raise ValueError(f"Duplicate image record ID or unresolved train image: {key}, {name}")\n        if (row.get("width"), row.get("height")) != (2048, 2048):\n            raise ValueError(f"Bad COCO dimensions: {key}")\n        records[key] = row\n        counts[key] = 0\n        annotations[key] = []\n    seen_annotations = set()\n    segmentation_types = {}\n    for row in data["annotations"]:\n        key, aid = str(row["image_id"]), str(row["id"])\n        if key not in records or aid in seen_annotations:\n            raise ValueError(f"Orphan or duplicate annotation: {aid}, image {key}")\n        seen_annotations.add(aid)\n        seg = row.get("segmentation")\n        if isinstance(seg, list) and seg:\n            for polygon in seg:\n                if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:\n                    raise ValueError(f"Invalid polygon structure: {aid}")\n                if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 2048 for v in polygon):\n                    raise ValueError(f"Nonfinite/out-of-bounds polygon: {aid}")\n            kind = "polygon"\n        elif isinstance(seg, dict) and seg.get("size") == [2048, 2048] and isinstance(seg.get("counts"), (list, str)):\n            kind = "rle"\n            # Validate every native RLE, rather than passing malformed buffers to C.\n            counts_value = seg["counts"]\n            if isinstance(counts_value, str):\n                counts_value = parse_compressed_counts(counts_value)\n            if not counts_value or any(type(v) is not int or v < 0 for v in counts_value) or sum(counts_value) != 2048 * 2048:\n                raise ValueError(f"Invalid RLE counts: {aid}")\n        else:\n            raise ValueError(f"Unsupported or empty segmentation: {aid}")\n        segmentation_types[kind] = segmentation_types.get(kind, 0) + 1\n        annotations[key].append(row)\n        counts[key] += 1\n    unreferenced = sorted(set(p for p, v in physical.items() if v["split"] == "train") - {Path(r["file_name"]).stem for r in records.values()})\n    if unreferenced:\n        raise ValueError(f"Train files absent from annotations: {unreferenced[:10]}")\n    # Candidate connected components join same-day observations and exact pixel duplicates.\n    parent = {p: p for p in physical}\n    def find(x):\n        while parent[x] != x:\n            parent[x] = parent[parent[x]]\n            x = parent[x]\n        return x\n    def union(a, b):\n        a, b = find(a), find(b)\n        parent[max(a, b)] = min(a, b)\n    days, hashes = {}, {}\n    for p, row in physical.items():\n        for value, groups in ((row["day"], days), (row["pixel_sha256"], hashes)):\n            if value in groups:\n                union(p, groups[value][0])\n            groups.setdefault(value, []).append(p)\n    fold_rows, manifest = [], []\n    for key, row in sorted(records.items()):\n        p = Path(row["file_name"]).stem\n        group = find(p)\n        fold = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5\n        fold_rows.append({"image_id": key, "physical_id": p, "group": group, "fold": fold})\n        manifest.append({"image_id": key, **physical[p], "annotation_count": counts[key], "license": row.get("license")})\n    write_csv(out / "manifest.csv", manifest)\n    write_csv(out / "folds_candidate.csv", fold_rows)\n    dump(out / "group_report.json", {"policy": "candidate only: connected same-day / exact pixel duplicate groups; not final CV",\n         "exact_duplicate_groups": [v for v in hashes.values() if len(v) > 1],\n         "multi_image_day_groups": {k: v for k, v in days.items() if len(v) > 1},\n         "physical_images": list(physical.values()),\n         "train_test_day_overlap": [k for k, v in days.items() if len({physical[p]["split"] for p in v}) > 1]})\n    audit.update({"annotation_file": str(annotation_path), "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),\n                  "physical_train": sum(v["split"] == "train" for v in physical.values()),\n                  "physical_test": sum(v["split"] == "test" for v in physical.values()),\n                  "annotator_image_records": len(records), "annotation_count": len(seen_annotations),\n                  "segmentation_types": segmentation_types, "licenses": data.get("licenses"),\n                  "fold_record_counts": {i: sum(r["fold"] == i for r in fold_rows) for i in range(5)},\n                  "audit_seconds": time.monotonic() - START})\n    dump(out / "audit.json", audit)\n    # One annotator\'s union mask per chosen physical image; this is only a plumbing test.\n    xs, ys, selected, chosen = [], [], [], set()\n    decoded_annotation_records = 0\n    for key, row in sorted(records.items()):\n        p = Path(row["file_name"]).stem\n        if p in chosen or counts[key] == 0:\n            continue\n        chosen.add(p)\n        mask = np.zeros((2048, 2048), dtype=np.uint8)\n        for ann in annotations[key]:\n            seg = ann["segmentation"]\n            if isinstance(seg, list):\n                rle = coco_mask.merge(coco_mask.frPyObjects(seg, 2048, 2048))\n            elif isinstance(seg["counts"], list):\n                rle = coco_mask.frPyObjects(seg, 2048, 2048)\n            else:\n                rle = {"size": [2048, 2048], "counts": seg["counts"].encode("ascii")}\n            decoded = coco_mask.decode(rle)\n            if decoded.shape != mask.shape:\n                raise ValueError(f"Decoded shape mismatch: {ann[\'id\']}")\n            mask |= decoded\n            decoded_annotation_records += 1\n        with Image.open(files[Path(row["file_name"]).name]) as im:\n            xs.append(np.asarray(im.convert("L").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255)\n        ys.append(np.asarray(Image.fromarray(mask).resize((256, 256), Image.Resampling.NEAREST), dtype=np.float32))\n        selected.append(key)\n        if len(xs) == 4:\n            break\n    if len(xs) < 2 or not any(np.any(y) for y in ys):\n        raise ValueError("Need at least two real annotated images and nonempty targets")\n    audit.update(decoded_annotation_records=decoded_annotation_records,\n                 decoded_annotator_image_records=len(selected))\n    audit["stages"].update(structural_audit="partial", mask_decoding="selected_records_only", training="running")\n    dump(out / "audit.json", audit)\n    torch.manual_seed(2026)\n    torch.cuda.manual_seed_all(2026)\n    model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1),\n                          nn.ReLU(), nn.Conv2d(8, 1, 1)).cuda()\n    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n    x = torch.from_numpy(np.stack(xs)[:, None]).cuda()\n    y = torch.from_numpy(np.stack(ys)[:, None]).cuda()\n    training_start = time.monotonic()\n    losses, step_seconds = [], []\n    for step in range(args.steps):\n        if time.monotonic() - START > args.timeout - 30:\n            raise TimeoutError("Stopped before hard timeout; insufficient training time")\n        tick = time.monotonic()\n        optimizer.zero_grad(set_to_none=True)\n        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)\n        if not torch.isfinite(loss):\n            raise ValueError("Nonfinite training loss")\n        loss.backward()\n        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):\n            raise ValueError("Nonfinite training gradients")\n        optimizer.step()\n        torch.cuda.synchronize()\n        losses.append(float(loss.detach().cpu()))\n        step_seconds.append(time.monotonic() - tick)\n    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},\n                "architecture": "original Conv1-8-8-1", "smoke_only": True, "training_image_ids": selected}, out / "tiny_checkpoint.pt")\n    audit["stages"].update(training="success", checkpoint="saved_not_reload_validated")\n    audit.update({"status": "environment_smoke_success", "training_ids": selected, "training_steps": len(losses), "losses": losses,\n                  "step_seconds": step_seconds, "train_seconds": time.monotonic() - training_start,\n                  "total_seconds": time.monotonic() - START, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),\n                  "test_inference_seconds": 0, "cv_score": None})\n    dump(out / "audit.json", audit)\n    print(json.dumps(audit), flush=True)\n\n\ndef parse_compressed_counts(value):\n    """Bounded Python COCO counts parser, before any native decoder."""\n    counts, pos = [], 0\n    while pos < len(value):\n        number, shift = 0, 0\n        while True:\n            if pos >= len(value) or shift > 30:\n                raise ValueError("Truncated or oversized compressed RLE")\n            char = ord(value[pos]) - 48\n            pos += 1\n            if not 0 <= char <= 63:\n                raise ValueError("Invalid compressed RLE character")\n            number |= (char & 31) << shift\n            shift += 5\n            if not char & 32:\n                if char & 16:\n                    number |= -1 << shift\n                break\n        if len(counts) > 2:\n            number += counts[-2]\n        if number < 0 or number > 2048 * 2048:\n            raise ValueError("RLE run out of bounds")\n        counts.append(number)\n        if len(counts) > 2048 * 2048 + 1:\n            raise ValueError("Too many RLE runs")\n    return counts\n\n\nif __name__ == "__main__":\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))\n    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))\n    parser.add_argument("--steps", type=int, default=10)\n    parser.add_argument("--timeout", type=int, default=360)\n    args = parser.parse_args()\n    if not 1 <= args.steps <= 20 or not 60 <= args.timeout <= 360:\n        parser.error("steps must be 1..20; timeout 60..360 (server limit must be 450)")\n    if not Path("/kaggle").is_dir() or args.input.resolve() != Path("/kaggle/input").resolve():\n        parser.error("This smoke test is Kaggle-only and must use /kaggle/input")\n    args.output.mkdir(parents=True, exist_ok=True)\n    def hard_stop():\n        details = {"status": "hard_timeout", "elapsed_seconds": time.monotonic() - START}\n        terminal_receipt(args.output, "hard_timeout", details)\n        dump(args.output / "timeout.json", details)\n        os._exit(124)\n    timer = threading.Timer(max(1, args.timeout - (time.monotonic() - START)), hard_stop)\n    timer.daemon = True\n    timer.start()\n    try:\n        main(args)\n    except Exception:\n        details = {"status": "failed", "elapsed_seconds": time.monotonic() - START,\n                   "traceback": traceback.format_exc()}\n        terminal_receipt(args.output, "failed", details)\n        dump(args.output / "failure.json", details)\n        raise\n    finally:\n        timer.cancel()\n', 'validate_submission.py': '"""Validate variable-row filament instance CSV against the actual test manifest."""\nimport argparse\nimport csv\nimport json\nfrom pathlib import Path, PurePosixPath\n\nfrom gpu_smoke_audit import parse_compressed_counts\n\n# Valid high-frequency 2048x2048 masks can exceed Python CSV\'s 128 KiB default.\nCSV_FIELD_LIMIT = 64 * 1024 * 1024\n\n\ndef test_ids_from_manifest(path):\n    manifest = json.loads(Path(path).read_text(encoding="utf-8"))\n    if manifest.get("complete") is not True:\n        raise ValueError("A complete official file manifest is required")\n    ids = []\n    for row in manifest["files"]:\n        file = PurePosixPath(row["name"])\n        if "test_images" in file.parts and file.suffix.lower() in {".jpg", ".jpeg"}:\n            ids.append(file.stem)\n    if not ids or len(ids) != len(set(ids)):\n        raise ValueError("Manifest test IDs are empty or ambiguous")\n    return set(ids)\n\n\ndef validate_rle(value):\n    from pycocotools import mask as coco_mask\n    if not value or value != value.strip() or value.startswith(("b\'", \'b"\', "{")):\n        raise ValueError("RLE must be the compressed counts string alone")\n    counts = parse_compressed_counts(value)\n    if not counts or sum(counts) != 2048 * 2048:\n        raise ValueError("RLE runs must cover exactly 2048 x 2048 pixels")\n    # The bounded parser above rejects malformed sizes before entering native code.\n    decoded = coco_mask.decode({"size": [2048, 2048], "counts": value.encode("ascii")})\n    if decoded.shape != (2048, 2048):\n        raise ValueError("RLE decoded dimensions do not match [2048, 2048]")\n    # Canonical roundtrip prevents accepting trailing garbage / alternate byte syntax.\n    canonical = coco_mask.encode(decoded)["counts"].decode("ascii")\n    if canonical != value:\n        raise ValueError("RLE is not canonical compressed COCO counts")\n    return int(decoded.sum())\n\n\ndef validate_submission(path, manifest=None, *, allowed_ids=None, reject_empty_masks=False, reject_all_empty=False):\n    csv.field_size_limit(max(csv.field_size_limit(), CSV_FIELD_LIMIT))\n    if allowed_ids is None:\n        if manifest is None:\n            manifest = Path(__file__).resolve().parents[1] / "research/data-file-manifest.json"\n        test_ids = test_ids_from_manifest(manifest)\n    else:\n        values = list(allowed_ids)\n        if not values or len(values) != len(set(values)) or any(not isinstance(v, str) or not v or v != v.strip() or \'_\' in v or any(c.isspace() for c in v) for v in values):\n            raise ValueError("Explicit allowed held-out IDs must be nonempty, unique physical-image stems")\n        test_ids = set(values)\n    seen, predicted, empty_masks, total_area = set(), set(), 0, 0\n    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:\n        reader = csv.DictReader(stream, strict=True)\n        if reader.fieldnames != ["filament_id", "segmentation_rle"]:\n            raise ValueError("Expected exactly filament_id,segmentation_rle in that order")\n        for line, row in enumerate(reader, 2):\n            try:\n                if None in row or None in row.values():\n                    raise ValueError("Wrong field count")\n                instance = row["filament_id"]\n                if instance in seen:\n                    raise ValueError(f"Duplicate instance ID: {instance}")\n                if instance != instance.strip() or "_" not in instance:\n                    raise ValueError("Instance ID needs original image stem and nonempty underscore suffix")\n                image, suffix = instance.split("_", 1)\n                if image not in test_ids or not suffix or any(c.isspace() for c in suffix):\n                    raise ValueError(f"Foreign image ID or invalid instance suffix: {instance}")\n                area = validate_rle(row["segmentation_rle"])\n                if reject_empty_masks and area == 0:\n                    raise ValueError("Candidate policy rejects empty instance masks")\n                empty_masks += area == 0\n                total_area += area\n                seen.add(instance)\n                predicted.add(image)\n            except (ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:\n                raise ValueError(f"CSV line {line}: {exc}") from exc\n    if reject_all_empty and total_area == 0:\n        raise ValueError("Candidate policy rejects all-empty output (including no rows)")\n    missing = sorted(test_ids - predicted)\n    warnings = []\n    if missing:\n        warnings.append(f"{len(missing)} test images have no predicted rows; server coverage/empty-image conventions remain unverified")\n    if empty_masks:\n        warnings.append(f"{empty_masks} empty instance masks; server acceptance remains unverified")\n    return {"valid": True, "rows": len(seen), "predicted_images": len(predicted),\n            "test_images": len(test_ids), "missing_image_ids": missing,\n            "empty_masks": empty_masks, "warnings": warnings,\n            "id_scope": "explicit held-out IDs" if allowed_ids is not None else "official test manifest",\n            "candidate_policy": {"reject_empty_masks": reject_empty_masks, "reject_all_empty": reject_all_empty},\n            "scope": "format validation only; no exact row-count, full coverage or server-score guarantee"}\n\n\nif __name__ == "__main__":\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("csv", type=Path)\n    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "research/data-file-manifest.json")\n    parser.add_argument("--allowed-heldout-ids", type=Path, help="JSON list of permitted physical image stems; replaces test manifest ID scope")\n    parser.add_argument("--reject-empty-masks", action="store_true")\n    parser.add_argument("--reject-all-empty", action="store_true")\n    args = parser.parse_args()\n    try:\n        ids = json.loads(args.allowed_heldout_ids.read_text(encoding="utf-8")) if args.allowed_heldout_ids else None\n        if ids is not None and not isinstance(ids, list):\n            raise ValueError("Allowed held-out IDs file must contain a JSON list")\n        print(json.dumps(validate_submission(args.csv, args.manifest, allowed_ids=ids,\n                                             reject_empty_masks=args.reject_empty_masks, reject_all_empty=args.reject_all_empty), indent=2))\n    except (ValueError, csv.Error, OSError) as exc:\n        parser.exit(1, f"Validation failed: {exc}\\n")\n', 'folds.csv': 'record_id,physical_id,group_id,fold\n010101-20110114105034Ch,20110114105034Ch,20110114105034Ch,1\n010101-20110203082634Lh,20110203082634Lh,20110203082634Lh,2\n010101-20110220082634Lh,20110220082634Lh,20110220082634Lh,2\n010101-20110327143554Bh,20110327143554Bh,20110324082614Th,2\n010101-20110625163814Mh,20110625163814Mh,20110625163814Mh,2\n010101-20110914063134Lh,20110914063134Lh,20110914063134Lh,1\n010101-20110918112834Ch,20110918112834Ch,20110918112834Ch,1\n010101-20111027145954Bh,20111027145954Bh,20111027145954Bh,2\n010101-20111029063154Uh,20111029063154Uh,20111029063154Uh,0\n010101-20120206063134Lh,20120206063134Lh,20120205172414Mh,2\n010101-20120215153654Bh,20120215153654Bh,20120215153654Bh,0\n010101-20120330063134Lh,20120330063134Lh,20120330063134Lh,0\n010101-20120427063154Uh,20120427063154Uh,20120427063154Uh,0\n010101-20120603123034Ch,20120603123034Ch,20120603123034Ch,2\n010101-20121109103634Ch,20121109103634Ch,20121108155954Bh,2\n010101-20130425135954Bh,20130425135954Bh,20130425135954Bh,2\n010101-20130811140254Bh,20130811140254Bh,20130807073114Th,2\n010101-20130903060834Lh,20130903060834Lh,20130903060834Lh,0\n010101-20131006075314Th,20131006075314Th,20131006075314Th,0\n010101-20131016185334Ch,20131016185334Ch,20131016185334Ch,1\n010101-20131017185334Ch,20131017185334Ch,20131016185334Ch,1\n010101-20131025185414Mh,20131025185414Mh,20131025185414Mh,0\n010101-20131109222434Lh,20131109222434Lh,20131107043054Uh,1\n010101-20131128192214Mh,20131128192214Mh,20131128192214Mh,1\n010101-20140127180134Ch,20140127180134Ch,20140127180134Ch,2\n010101-20140312081114Th,20140312081114Th,20140312025454Uh,0\n010101-20140321195814Mh,20140321195814Mh,20140321195814Mh,1\n010101-20140509195814Mh,20140509195814Mh,20140508195834Ch,1\n010101-20140513012554Uh,20140513012554Uh,20140513012554Uh,2\n010101-20140604195854Bh,20140604195854Bh,20140603070414Th,0\n010101-20140813064254Uh,20140813064254Uh,20140813064254Uh,2\n010101-20141027025054Uh,20141027025054Uh,20141024195834Ch,2\n010101-20141027224834Lh,20141027224834Lh,20141024195834Ch,2\n010101-20141129222134Lh,20141129222134Lh,20141129222134Lh,1\n010101-20141210201814Mh,20141210201814Mh,20141210201814Mh,0\n010101-20150112081954Uh,20150112081954Uh,20150112081954Uh,0\n010101-20150518145654Bh,20150518145654Bh,20150518145654Bh,1\n010101-20150527161214Mh,20150527161214Mh,20150527082014Th,2\n010101-20150531082034Lh,20150531082034Lh,20150531082034Lh,1\n010101-20150626161514Mh,20150626161514Mh,20150625082014Th,0\n010101-20150806082034Lh,20150806082034Lh,20150806082034Lh,0\n010101-20150927111834Ch,20150927111834Ch,20150926111934Ch,2\n010101-20151117172314Mh,20151117172314Mh,20151116201114Mh,1\n010101-20151122172614Mh,20151122172614Mh,20151122172614Mh,2\n010101-20160226173214Mh,20160226173214Mh,20160225173314Mh,0\n010101-20160320133714Th,20160320133714Th,20160320133714Th,0\n010101-20160401170314Mh,20160401170314Mh,20160401133734Ch,0\n010101-20160526011254Uh,20160526011254Uh,20160526011254Uh,2\n010101-20160606163514Mh,20160606163514Mh,20160606163514Mh,2\n010101-20160703133714Th,20160703133714Th,20160703133714Th,0\n010101-20160714235434Lh,20160714235434Lh,20160714235434Lh,1\n010101-20161129022454Uh,20161129022454Uh,20161129022454Uh,0\n010101-20161222133714Th,20161222133714Th,20161222133714Th,1\n010101-20161224024154Uh,20161224024154Uh,20161222133714Th,1\n010101-20170103030554Uh,20170103030554Uh,20170102024454Uh,0\n010101-20170107133714Th,20170107133714Th,20170106133714Th,2\n010101-20170218173710Mh,20170218173710Mh,20170217165250Bh,0\n010101-20170418013450Uh,20170418013450Uh,20170417014150Uh,1\n010101-20171008144250Bh,20171008144250Bh,20171007133710Th,1\n010101-20171011170310Mh,20171011170310Mh,20171011170310Mh,1\n010101-20171027223030Lh,20171027223030Lh,20171027223030Lh,0\n010101-20171101133730Ch,20171101133730Ch,20171101133730Ch,2\n010101-20180309000910Mh,20180309000910Mh,20180309000910Mh,0\n010101-20180318020350Uh,20180318020350Uh,20180318020350Uh,2\n010101-20180825205030Ch,20180825205030Ch,20180825205030Ch,0\n010101-20181023232430Lh,20181023232430Lh,20181023232430Lh,1\n010101-20190602025430Lh,20190602025430Lh,20190602025430Lh,1\n010101-20190626025430Lh,20190626025430Lh,20190626025430Lh,2\n010101-20200110073230Lh,20200110073230Lh,20200110073230Lh,1\n010101-20200116073250Uh,20200116073250Uh,20200116073250Uh,2\n010101-20210222222850Bh,20210222222850Bh,20210222222850Bh,1\n010101-20211016145050Bh,20211016145050Bh,20211016145050Bh,2\n010102-20110203082634Lh,20110203082634Lh,20110203082634Lh,2\n010102-20110220082634Lh,20110220082634Lh,20110220082634Lh,2\n010102-20110327143554Bh,20110327143554Bh,20110324082614Th,2\n010102-20110914063134Lh,20110914063134Lh,20110914063134Lh,1\n010102-20110918112834Ch,20110918112834Ch,20110918112834Ch,1\n010102-20111027145954Bh,20111027145954Bh,20111027145954Bh,2\n010102-20111029063154Uh,20111029063154Uh,20111029063154Uh,0\n010102-20120206063134Lh,20120206063134Lh,20120205172414Mh,2\n010102-20120215153654Bh,20120215153654Bh,20120215153654Bh,0\n010102-20120330063134Lh,20120330063134Lh,20120330063134Lh,0\n010102-20120427063154Uh,20120427063154Uh,20120427063154Uh,0\n010102-20120603123034Ch,20120603123034Ch,20120603123034Ch,2\n010102-20121109103634Ch,20121109103634Ch,20121108155954Bh,2\n010102-20130425135954Bh,20130425135954Bh,20130425135954Bh,2\n010102-20130811140254Bh,20130811140254Bh,20130807073114Th,2\n010102-20130903060834Lh,20130903060834Lh,20130903060834Lh,0\n010102-20131006075314Th,20131006075314Th,20131006075314Th,0\n010102-20131016185334Ch,20131016185334Ch,20131016185334Ch,1\n010102-20131017185334Ch,20131017185334Ch,20131016185334Ch,1\n010102-20131025185414Mh,20131025185414Mh,20131025185414Mh,0\n010102-20131109222434Lh,20131109222434Lh,20131107043054Uh,1\n010102-20131128192214Mh,20131128192214Mh,20131128192214Mh,1\n010102-20140112180134Ch,20140112180134Ch,20140111224134Lh,1\n010102-20140127180134Ch,20140127180134Ch,20140127180134Ch,2\n010102-20140312081114Th,20140312081114Th,20140312025454Uh,0\n010102-20140321195814Mh,20140321195814Mh,20140321195814Mh,1\n010102-20140509195814Mh,20140509195814Mh,20140508195834Ch,1\n010102-20140513012554Uh,20140513012554Uh,20140513012554Uh,2\n010102-20140604195854Bh,20140604195854Bh,20140603070414Th,0\n010102-20140813064254Uh,20140813064254Uh,20140813064254Uh,2\n010102-20141027025054Uh,20141027025054Uh,20141024195834Ch,2\n010102-20141027224834Lh,20141027224834Lh,20141024195834Ch,2\n010102-20141129222134Lh,20141129222134Lh,20141129222134Lh,1\n010102-20141210201814Mh,20141210201814Mh,20141210201814Mh,0\n010102-20150112081954Uh,20150112081954Uh,20150112081954Uh,0\n010102-20150518145654Bh,20150518145654Bh,20150518145654Bh,1\n010102-20150527161214Mh,20150527161214Mh,20150527082014Th,2\n010102-20150531082034Lh,20150531082034Lh,20150531082034Lh,1\n010102-20150626161514Mh,20150626161514Mh,20150625082014Th,0\n010102-20150806082034Lh,20150806082034Lh,20150806082034Lh,0\n010102-20150927111834Ch,20150927111834Ch,20150926111934Ch,2\n010102-20151117172314Mh,20151117172314Mh,20151116201114Mh,1\n010102-20151122172614Mh,20151122172614Mh,20151122172614Mh,2\n010102-20160226173214Mh,20160226173214Mh,20160225173314Mh,0\n010102-20160320133714Th,20160320133714Th,20160320133714Th,0\n010102-20160401170314Mh,20160401170314Mh,20160401133734Ch,0\n010102-20160526011254Uh,20160526011254Uh,20160526011254Uh,2\n010102-20160606163514Mh,20160606163514Mh,20160606163514Mh,2\n010102-20160703133714Th,20160703133714Th,20160703133714Th,0\n010102-20160714235434Lh,20160714235434Lh,20160714235434Lh,1\n010102-20161129022454Uh,20161129022454Uh,20161129022454Uh,0\n010102-20161222133714Th,20161222133714Th,20161222133714Th,1\n010102-20161224024154Uh,20161224024154Uh,20161222133714Th,1\n010102-20170103030554Uh,20170103030554Uh,20170102024454Uh,0\n010102-20170218173710Mh,20170218173710Mh,20170217165250Bh,0\n010102-20170418013450Uh,20170418013450Uh,20170417014150Uh,1\n010102-20171008144250Bh,20171008144250Bh,20171007133710Th,1\n010102-20171011170310Mh,20171011170310Mh,20171011170310Mh,1\n010102-20171027223030Lh,20171027223030Lh,20171027223030Lh,0\n010102-20171101133730Ch,20171101133730Ch,20171101133730Ch,2\n010102-20180318020350Uh,20180318020350Uh,20180318020350Uh,2\n010102-20180825205030Ch,20180825205030Ch,20180825205030Ch,0\n010102-20190602025430Lh,20190602025430Lh,20190602025430Lh,1\n010102-20190626025430Lh,20190626025430Lh,20190626025430Lh,2\n010102-20200110073230Lh,20200110073230Lh,20200110073230Lh,1\n010103-20110114105034Ch,20110114105034Ch,20110114105034Ch,1\n010103-20110203082634Lh,20110203082634Lh,20110203082634Lh,2\n010103-20110914063134Lh,20110914063134Lh,20110914063134Lh,1\n010103-20111027145954Bh,20111027145954Bh,20111027145954Bh,2\n010103-20120603123034Ch,20120603123034Ch,20120603123034Ch,2\n010103-20121109103634Ch,20121109103634Ch,20121108155954Bh,2\n010103-20131016185334Ch,20131016185334Ch,20131016185334Ch,1\n010103-20140112180134Ch,20140112180134Ch,20140111224134Lh,1\n010103-20140509195814Mh,20140509195814Mh,20140508195834Ch,1\n010103-20140604195854Bh,20140604195854Bh,20140603070414Th,0\n010103-20150112081954Uh,20150112081954Uh,20150112081954Uh,0\n010103-20150927111834Ch,20150927111834Ch,20150926111934Ch,2\n010103-20151117172314Mh,20151117172314Mh,20151116201114Mh,1\n010103-20151122172614Mh,20151122172614Mh,20151122172614Mh,2\n010103-20160320133714Th,20160320133714Th,20160320133714Th,0\n010103-20160401170314Mh,20160401170314Mh,20160401133734Ch,0\n010103-20160526011254Uh,20160526011254Uh,20160526011254Uh,2\n010103-20160703133714Th,20160703133714Th,20160703133714Th,0\n010103-20161224024154Uh,20161224024154Uh,20161222133714Th,1\n010103-20170103030554Uh,20170103030554Uh,20170102024454Uh,0\n010103-20170418013450Uh,20170418013450Uh,20170417014150Uh,1\n010103-20171011170310Mh,20171011170310Mh,20171011170310Mh,1\n010103-20171101133730Ch,20171101133730Ch,20171101133730Ch,2\n010103-20180309000910Mh,20180309000910Mh,20180309000910Mh,0\n010103-20180318020350Uh,20180318020350Uh,20180318020350Uh,2\n010103-20180825205030Ch,20180825205030Ch,20180825205030Ch,0\n010103-20190626025430Lh,20190626025430Lh,20190626025430Lh,2\n010103-20200110073230Lh,20200110073230Lh,20200110073230Lh,1\n010103-20210222222850Bh,20210222222850Bh,20210222222850Bh,1\n010201-20110724164614Mh,20110724164614Mh,20110724164614Mh,0\n010201-20110823120334Ch,20110823120334Ch,20110823120334Ch,1\n010201-20120729122634Ch,20120729122634Ch,20120729122634Ch,2\n010201-20120910074014Th,20120910074014Th,20120910074014Th,1\n010201-20120917142654Bh,20120917142654Bh,20120917142654Bh,1\n010201-20121006063534Lh,20121006063534Lh,20121006063534Lh,0\n010201-20130717123434Ch,20130717123434Ch,20130716162214Mh,1\n010201-20131224192154Bh,20131224192154Bh,20131224192154Bh,0\n010201-20140225163314Th,20140225163314Th,20140225163314Th,0\n010201-20140414020354Uh,20140414020354Uh,20140413195914Mh,2\n010201-20141012015654Uh,20141012015654Uh,20141012015654Uh,1\n010201-20141016022554Uh,20141016022554Uh,20141016022554Uh,1\n010201-20150819082014Th,20150819082014Th,20150819082014Th,0\n010201-20170502233130Lh,20170502233130Lh,20170502233130Lh,1\n010201-20170726235030Lh,20170726235030Lh,20170726235030Lh,0\n010201-20171103222630Lh,20171103222630Lh,20171103222630Lh,2\n010201-20180804234530Lh,20180804234530Lh,20180804234530Lh,1\n010201-20190611032810Mh,20190611032810Mh,20190611032810Mh,1\n010201-20210201020330Lh,20210201020330Lh,20210201020330Lh,2\n010201-20211225025850Uh,20211225025850Uh,20211225025850Uh,1\n010202-20110823120334Ch,20110823120334Ch,20110823120334Ch,1\n010202-20120729122634Ch,20120729122634Ch,20120729122634Ch,2\n010202-20120910074014Th,20120910074014Th,20120910074014Th,1\n010202-20120917142654Bh,20120917142654Bh,20120917142654Bh,1\n010202-20121006063534Lh,20121006063534Lh,20121006063534Lh,0\n010202-20121024164714Mh,20121024164714Mh,20121024164714Mh,0\n010202-20130807073114Th,20130807073114Th,20130807073114Th,2\n010202-20130903141654Bh,20130903141654Bh,20130903060834Lh,0\n010202-20140111224134Lh,20140111224134Lh,20140111224134Lh,1\n010202-20140225163314Th,20140225163314Th,20140225163314Th,0\n010202-20140414020354Uh,20140414020354Uh,20140413195914Mh,2\n010202-20140428233234Lh,20140428233234Lh,20140428233234Lh,0\n010202-20140531195834Ch,20140531195834Ch,20140531195834Ch,1\n010202-20140615195854Bh,20140615195854Bh,20140615195814Mh,0\n010202-20140918195854Bh,20140918195854Bh,20140918195854Bh,2\n010202-20141004075414Th,20141004075414Th,20141004075414Th,0\n010202-20141012015654Uh,20141012015654Uh,20141012015654Uh,1\n010202-20141016022554Uh,20141016022554Uh,20141016022554Uh,1\n010202-20150716162114Mh,20150716162114Mh,20150716162114Mh,0\n010202-20150819082014Th,20150819082014Th,20150819082014Th,0\n010202-20160223173514Mh,20160223173514Mh,20160223173514Mh,2\n010202-20161110151454Bh,20161110151454Bh,20161108021454Uh,0\n010202-20161231133734Ch,20161231133734Ch,20161231133734Ch,0\n010202-20170222133710Th,20170222133710Th,20170222133710Th,2\n010202-20170327143550Bh,20170327143550Bh,20170327143550Bh,0\n010202-20170329133730Ch,20170329133730Ch,20170329133730Ch,2\n010202-20170502233130Lh,20170502233130Lh,20170502233130Lh,1\n010202-20170719235330Lh,20170719235330Lh,20170719235330Lh,1\n010202-20170726235030Lh,20170726235030Lh,20170726235030Lh,0\n010202-20171004143950Bh,20171004143950Bh,20171004143950Bh,2\n010202-20171103222630Lh,20171103222630Lh,20171103222630Lh,2\n010202-20180112130330Ch,20180112130330Ch,20180112130330Ch,1\n010202-20180804234530Lh,20180804234530Lh,20180804234530Lh,1\n010202-20190611032810Mh,20190611032810Mh,20190611032810Mh,1\n010202-20210201020330Lh,20210201020330Lh,20210201020330Lh,2\n010202-20211225025850Uh,20211225025850Uh,20211225025850Uh,1\n010202-20220617185352Mh,20220617185352Mh,20220617185352Mh,2\n010203-20110823120334Ch,20110823120334Ch,20110823120334Ch,1\n010203-20120729122634Ch,20120729122634Ch,20120729122634Ch,2\n010203-20120910074014Th,20120910074014Th,20120910074014Th,1\n010203-20120917142654Bh,20120917142654Bh,20120917142654Bh,1\n010203-20130208063154Uh,20130208063154Uh,20130208063154Uh,1\n010203-20130807073114Th,20130807073114Th,20130807073114Th,2\n010203-20130810121534Ch,20130810121534Ch,20130807073114Th,2\n010203-20130903141654Bh,20130903141654Bh,20130903060834Lh,0\n010203-20130919060834Lh,20130919060834Lh,20130919060834Lh,1\n010203-20140409081614Th,20140409081614Th,20140409081614Th,2\n010203-20140531195834Ch,20140531195834Ch,20140531195834Ch,1\n010203-20141004075414Th,20141004075414Th,20141004075414Th,0\n010203-20141012015654Uh,20141012015654Uh,20141012015654Uh,1\n010203-20141016022554Uh,20141016022554Uh,20141016022554Uh,1\n010203-20141022223334Lh,20141022223334Lh,20141022223334Lh,0\n010203-20150716162114Mh,20150716162114Mh,20150716162114Mh,0\n010203-20150819082014Th,20150819082014Th,20150819082014Th,0\n010203-20160223173514Mh,20160223173514Mh,20160223173514Mh,2\n010203-20161105222634Lh,20161105222634Lh,20161105222634Lh,0\n010203-20161110151454Bh,20161110151454Bh,20161108021454Uh,0\n010203-20161231133734Ch,20161231133734Ch,20161231133734Ch,0\n010203-20170106133714Th,20170106133714Th,20170106133714Th,2\n010203-20170222133710Th,20170222133710Th,20170222133710Th,2\n010203-20170310133730Ch,20170310133730Ch,20170310133730Ch,1\n010203-20170327143550Bh,20170327143550Bh,20170327143550Bh,0\n010203-20170329133730Ch,20170329133730Ch,20170329133730Ch,2\n010203-20170502233130Lh,20170502233130Lh,20170502233130Lh,1\n010203-20170726235030Lh,20170726235030Lh,20170726235030Lh,0\n010203-20171004143950Bh,20171004143950Bh,20171004143950Bh,2\n010203-20180112130330Ch,20180112130330Ch,20180112130330Ch,1\n010203-20180804234530Lh,20180804234530Lh,20180804234530Lh,1\n010203-20190611032810Mh,20190611032810Mh,20190611032810Mh,1\n010203-20210201020330Lh,20210201020330Lh,20210201020330Lh,2\n010203-20210621133650Bh,20210621133650Bh,20210620163530Mh,2\n010203-20211225025850Uh,20211225025850Uh,20211225025850Uh,1\n010203-20211229120210Th,20211229120210Th,20211229120130Ch,0\n010203-20220710085152Th,20220710085152Th,20220710085152Th,1\n010301-20110213093414Th,20110213093414Th,20110213093414Th,2\n010301-20120803102114Th,20120803102114Th,20120802090314Th,0\n010301-20121027063134Lh,20121027063134Lh,20121026063134Lh,2\n010301-20140605195854Bh,20140605195854Bh,20140603070414Th,0\n010301-20151119133734Ch,20151119133734Ch,20151116201114Mh,1\n010301-20160315133714Th,20160315133714Th,20160315133714Th,1\n010301-20160401133734Ch,20160401133734Ch,20160401133734Ch,0\n010301-20180805232510Mh,20180805232510Mh,20180804234530Lh,1\n010301-20181024232430Lh,20181024232430Lh,20181023232430Lh,1\n010301-20190609123330Ch,20190609123330Ch,20190609123330Ch,0\n010302-20110213093414Th,20110213093414Th,20110213093414Th,2\n010302-20111223085314Th,20111223085314Th,20111222063154Uh,2\n010302-20120803102114Th,20120803102114Th,20120802090314Th,0\n010302-20121027063134Lh,20121027063134Lh,20121026063134Lh,2\n010302-20130809072814Th,20130809072814Th,20130807073114Th,2\n010302-20131125192154Bh,20131125192154Bh,20131125192154Bh,1\n010302-20140605195854Bh,20140605195854Bh,20140603070414Th,0\n010302-20151119133734Ch,20151119133734Ch,20151116201114Mh,1\n010302-20151213023454Uh,20151213023454Uh,20151212171154Bh,1\n010302-20160204133734Ch,20160204133734Ch,20160204133734Ch,2\n010302-20160315133714Th,20160315133714Th,20160315133714Th,1\n010302-20170831170750Bh,20170831170750Bh,20170831170750Bh,1\n010302-20180805232510Mh,20180805232510Mh,20180804234530Lh,1\n010302-20180926014350Uh,20180926014350Uh,20180923112330Ch,0\n010302-20190609123330Ch,20190609123330Ch,20190609123330Ch,0\n010302-20210624133950Bh,20210624133950Bh,20210624133950Bh,0\n010303-20160204133734Ch,20160204133734Ch,20160204133734Ch,2\n010303-20170415145450Bh,20170415145450Bh,20170415145450Bh,1\n010303-20170831170750Bh,20170831170750Bh,20170831170750Bh,1\n010303-20180926014350Uh,20180926014350Uh,20180923112330Ch,0\n010303-20210624133950Bh,20210624133950Bh,20210624133950Bh,0\n010401-20110128174814Mh,20110128174814Mh,20110128174814Mh,0\n010401-20110215083814Th,20110215083814Th,20110215083814Th,1\n010401-20110301082654Uh,20110301082654Uh,20110301082654Uh,0\n010401-20110306082634Lh,20110306082634Lh,20110306082634Lh,0\n010401-20110410141654Bh,20110410141654Bh,20110410141654Bh,2\n010401-20110803164914Mh,20110803164914Mh,20110803164914Mh,2\n010401-20110923165814Mh,20110923165814Mh,20110923112234Ch,2\n010401-20111009110234Ch,20111009110234Ch,20111009110234Ch,1\n010401-20111011112054Uh,20111011112054Uh,20111011112054Uh,0\n010401-20111215063134Lh,20111215063134Lh,20111214063134Lh,1\n010401-20111222063154Uh,20111222063154Uh,20111222063154Uh,2\n010401-20111229063154Uh,20111229063154Uh,20111229063154Uh,1\n010401-20120104063134Lh,20120104063134Lh,20120103104134Ch,1\n010401-20120131063154Uh,20120131063154Uh,20120131063154Uh,0\n010401-20120205172414Mh,20120205172414Mh,20120205172414Mh,2\n010401-20120404063154Uh,20120404063154Uh,20120403083114Th,2\n010401-20120424072414Th,20120424072414Th,20120424072414Th,1\n010401-20120519070814Th,20120519070814Th,20120519070814Th,1\n010401-20120610175114Mh,20120610175114Mh,20120610175114Mh,1\n010401-20120703070814Th,20120703070814Th,20120703070814Th,2\n010401-20120723095014Th,20120723095014Th,20120723095014Th,2\n010401-20120802090314Th,20120802090314Th,20120802090314Th,0\n010401-20120913165314Mh,20120913165314Mh,20120913165314Mh,2\n010401-20121113103634Ch,20121113103634Ch,20121113103634Ch,0\n010401-20121129063154Uh,20121129063154Uh,20121129063154Uh,1\n010401-20121218063534Lh,20121218063534Lh,20121218063534Lh,2\n010401-20130221063134Lh,20130221063134Lh,20130221063134Lh,2\n010401-20130303063154Uh,20130303063154Uh,20130303063134Lh,0\n010401-20130319144654Bh,20130319144654Bh,20130319144654Bh,1\n010401-20130517063134Lh,20130517063134Lh,20130517063134Lh,1\n010401-20130813060834Lh,20130813060834Lh,20130813060834Lh,2\n010401-20130925143254Bh,20130925143254Bh,20130925143254Bh,1\n010401-20131007075314Th,20131007075314Th,20131006075314Th,0\n010401-20131011195414Mh,20131011195414Mh,20131011195414Mh,0\n010401-20131027025654Uh,20131027025654Uh,20131027025654Uh,2\n010401-20131107043054Uh,20131107043054Uh,20131107043054Uh,1\n010401-20140205180154Bh,20140205180154Bh,20140205180154Bh,2\n010401-20140216230134Lh,20140216230134Lh,20140216230134Lh,0\n010401-20140406195854Bh,20140406195854Bh,20140406195854Bh,0\n010401-20140413195914Mh,20140413195914Mh,20140413195914Mh,2\n010401-20140509071314Th,20140509071314Th,20140508195834Ch,1\n010401-20141006224634Lh,20141006224634Lh,20141006224634Lh,0\n010401-20141016195834Ch,20141016195834Ch,20141016022554Uh,1\n010401-20141106195854Bh,20141106195854Bh,20141105195814Mh,2\n010401-20141112195854Bh,20141112195854Bh,20141112195854Bh,0\n010401-20150125172714Mh,20150125172714Mh,20150125172714Mh,1\n010401-20150221155154Bh,20150221155154Bh,20150221155154Bh,2\n010401-20150228082414Th,20150228082414Th,20150228082414Th,0\n010401-20150308081954Uh,20150308081954Uh,20150308081954Uh,2\n010401-20150325015554Uh,20150325015554Uh,20150323172814Mh,2\n010401-20150423120034Ch,20150423120034Ch,20150423120034Ch,0\n010401-20150427162314Mh,20150427162314Mh,20150427162314Mh,1\n010401-20150614162514Mh,20150614162514Mh,20150614162514Mh,1\n010401-20150625082034Lh,20150625082034Lh,20150625082014Th,0\n010401-20150825082034Lh,20150825082034Lh,20150825082034Lh,1\n010401-20160213153054Bh,20160213153054Bh,20160213153054Bh,1\n010401-20160215133734Ch,20160215133734Ch,20160215133734Ch,0\n010401-20160225173314Mh,20160225173314Mh,20160225173314Mh,0\n010401-20160229151254Bh,20160229151254Bh,20160229151254Bh,1\n010401-20160403170114Mh,20160403170114Mh,20160401133734Ch,0\n010401-20160810234034Lh,20160810234034Lh,20160808133714Th,0\n010401-20160920230134Lh,20160920230134Lh,20160920230134Lh,2\n010401-20161205133714Th,20161205133714Th,20161205133714Th,0\n010401-20170119031254Uh,20170119031254Uh,20170119031254Uh,2\n010401-20170804140650Bh,20170804140650Bh,20170804140650Bh,1\n010401-20170927143450Bh,20170927143450Bh,20170927143450Bh,0\n010401-20171029150150Bh,20171029150150Bh,20171029150150Bh,0\n010401-20180306081810Th,20180306081810Th,20180306081810Th,2\n010401-20190412025430Lh,20190412025430Lh,20190412025430Lh,2\n010401-20201227202710Mh,20201227202710Mh,20201227202710Mh,1\n010401-20210223222830Ch,20210223222830Ch,20210222222850Bh,1\n010401-20210817165030Mh,20210817165030Mh,20210817165030Mh,2\n010401-20220721185332Bh,20220721185332Bh,20220721185332Bh,1\n010402-20110128174814Mh,20110128174814Mh,20110128174814Mh,0\n010402-20110215083814Th,20110215083814Th,20110215083814Th,1\n010402-20110301082654Uh,20110301082654Uh,20110301082654Uh,0\n010402-20110306082634Lh,20110306082634Lh,20110306082634Lh,0\n010402-20110410141654Bh,20110410141654Bh,20110410141654Bh,2\n010402-20110803164914Mh,20110803164914Mh,20110803164914Mh,2\n010402-20110923165814Mh,20110923165814Mh,20110923112234Ch,2\n010402-20111009110234Ch,20111009110234Ch,20111009110234Ch,1\n010402-20111215063134Lh,20111215063134Lh,20111214063134Lh,1\n010402-20111222063154Uh,20111222063154Uh,20111222063154Uh,2\n010402-20111229063154Uh,20111229063154Uh,20111229063154Uh,1\n010402-20120131063154Uh,20120131063154Uh,20120131063154Uh,0\n010402-20120205172414Mh,20120205172414Mh,20120205172414Mh,2\n010402-20120404063154Uh,20120404063154Uh,20120403083114Th,2\n010402-20120424072414Th,20120424072414Th,20120424072414Th,1\n010402-20120519070814Th,20120519070814Th,20120519070814Th,1\n010402-20120610175114Mh,20120610175114Mh,20120610175114Mh,1\n010402-20120703070814Th,20120703070814Th,20120703070814Th,2\n010402-20120723095014Th,20120723095014Th,20120723095014Th,2\n010402-20120802090314Th,20120802090314Th,20120802090314Th,0\n010402-20120913165314Mh,20120913165314Mh,20120913165314Mh,2\n010402-20121009165914Mh,20121009165914Mh,20121009165914Mh,0\n010402-20121113103634Ch,20121113103634Ch,20121113103634Ch,0\n010402-20121129063154Uh,20121129063154Uh,20121129063154Uh,1\n010402-20121218063534Lh,20121218063534Lh,20121218063534Lh,2\n010402-20130221063134Lh,20130221063134Lh,20130221063134Lh,2\n010402-20130303063154Uh,20130303063154Uh,20130303063134Lh,0\n010402-20130319144654Bh,20130319144654Bh,20130319144654Bh,1\n010402-20130429063154Uh,20130429063154Uh,20130429063154Uh,2\n010402-20130517063134Lh,20130517063134Lh,20130517063134Lh,1\n010402-20130813060834Lh,20130813060834Lh,20130813060834Lh,2\n010402-20130925143254Bh,20130925143254Bh,20130925143254Bh,1\n010402-20131007075314Th,20131007075314Th,20131006075314Th,0\n010402-20131011195414Mh,20131011195414Mh,20131011195414Mh,0\n010402-20131027025654Uh,20131027025654Uh,20131027025654Uh,2\n010402-20131107043054Uh,20131107043054Uh,20131107043054Uh,1\n010402-20140205180154Bh,20140205180154Bh,20140205180154Bh,2\n010402-20140216230134Lh,20140216230134Lh,20140216230134Lh,0\n010402-20140406195854Bh,20140406195854Bh,20140406195854Bh,0\n010402-20140509071314Th,20140509071314Th,20140508195834Ch,1\n010402-20141006224634Lh,20141006224634Lh,20141006224634Lh,0\n010402-20141016195834Ch,20141016195834Ch,20141016022554Uh,1\n010402-20141106195854Bh,20141106195854Bh,20141105195814Mh,2\n010402-20141112195854Bh,20141112195854Bh,20141112195854Bh,0\n010402-20150125172714Mh,20150125172714Mh,20150125172714Mh,1\n010402-20150221155154Bh,20150221155154Bh,20150221155154Bh,2\n010402-20150228082414Th,20150228082414Th,20150228082414Th,0\n010402-20150308081954Uh,20150308081954Uh,20150308081954Uh,2\n010402-20150325015554Uh,20150325015554Uh,20150323172814Mh,2\n010402-20150423120034Ch,20150423120034Ch,20150423120034Ch,0\n010402-20150427162314Mh,20150427162314Mh,20150427162314Mh,1\n010402-20150614162514Mh,20150614162514Mh,20150614162514Mh,1\n010402-20150625082034Lh,20150625082034Lh,20150625082014Th,0\n010402-20150825082034Lh,20150825082034Lh,20150825082034Lh,1\n010402-20160213153054Bh,20160213153054Bh,20160213153054Bh,1\n010402-20160215133734Ch,20160215133734Ch,20160215133734Ch,0\n010402-20160225173314Mh,20160225173314Mh,20160225173314Mh,0\n010402-20160229151254Bh,20160229151254Bh,20160229151254Bh,1\n010402-20160403170114Mh,20160403170114Mh,20160401133734Ch,0\n010402-20160430133714Th,20160430133714Th,20160430133714Th,0\n010402-20160810234034Lh,20160810234034Lh,20160808133714Th,0\n010402-20160920230134Lh,20160920230134Lh,20160920230134Lh,2\n010402-20161205133714Th,20161205133714Th,20161205133714Th,0\n010402-20170119031254Uh,20170119031254Uh,20170119031254Uh,2\n010402-20170804140650Bh,20170804140650Bh,20170804140650Bh,1\n010402-20170927143450Bh,20170927143450Bh,20170927143450Bh,0\n010402-20171007133730Ch,20171007133730Ch,20171007133710Th,1\n010402-20171029150150Bh,20171029150150Bh,20171029150150Bh,0\n010402-20180306081810Th,20180306081810Th,20180306081810Th,2\n010402-20190412025430Lh,20190412025430Lh,20190412025430Lh,2\n010402-20201227202710Mh,20201227202710Mh,20201227202710Mh,1\n010402-20210223222830Ch,20210223222830Ch,20210222222850Bh,1\n010402-20210817165030Mh,20210817165030Mh,20210817165030Mh,2\n010402-20220418020950Uh,20220418020950Uh,20220418020950Uh,2\n010402-20220721185332Bh,20220721185332Bh,20220721185332Bh,1\n010403-20110128174814Mh,20110128174814Mh,20110128174814Mh,0\n010403-20110215083814Th,20110215083814Th,20110215083814Th,1\n010403-20110301082654Uh,20110301082654Uh,20110301082654Uh,0\n010403-20110306082634Lh,20110306082634Lh,20110306082634Lh,0\n010403-20110410141654Bh,20110410141654Bh,20110410141654Bh,2\n010403-20110803164914Mh,20110803164914Mh,20110803164914Mh,2\n010403-20110923165814Mh,20110923165814Mh,20110923112234Ch,2\n010403-20111009110234Ch,20111009110234Ch,20111009110234Ch,1\n010403-20111215063134Lh,20111215063134Lh,20111214063134Lh,1\n010403-20111222063154Uh,20111222063154Uh,20111222063154Uh,2\n010403-20111229063154Uh,20111229063154Uh,20111229063154Uh,1\n010403-20120104063134Lh,20120104063134Lh,20120103104134Ch,1\n010403-20120131063154Uh,20120131063154Uh,20120131063154Uh,0\n010403-20120205172414Mh,20120205172414Mh,20120205172414Mh,2\n010403-20120404063154Uh,20120404063154Uh,20120403083114Th,2\n010403-20120424072414Th,20120424072414Th,20120424072414Th,1\n010403-20120519070814Th,20120519070814Th,20120519070814Th,1\n010403-20120610175114Mh,20120610175114Mh,20120610175114Mh,1\n010403-20120703070814Th,20120703070814Th,20120703070814Th,2\n010403-20120723095014Th,20120723095014Th,20120723095014Th,2\n010403-20120802090314Th,20120802090314Th,20120802090314Th,0\n010403-20120913165314Mh,20120913165314Mh,20120913165314Mh,2\n010403-20121009165914Mh,20121009165914Mh,20121009165914Mh,0\n010403-20121113103634Ch,20121113103634Ch,20121113103634Ch,0\n010403-20121129063154Uh,20121129063154Uh,20121129063154Uh,1\n010403-20121218063534Lh,20121218063534Lh,20121218063534Lh,2\n010403-20130221063134Lh,20130221063134Lh,20130221063134Lh,2\n010403-20130303063154Uh,20130303063154Uh,20130303063134Lh,0\n010403-20130319144654Bh,20130319144654Bh,20130319144654Bh,1\n010403-20130429063154Uh,20130429063154Uh,20130429063154Uh,2\n010403-20130517063134Lh,20130517063134Lh,20130517063134Lh,1\n010403-20130813060834Lh,20130813060834Lh,20130813060834Lh,2\n010403-20130925143254Bh,20130925143254Bh,20130925143254Bh,1\n010403-20131007075314Th,20131007075314Th,20131006075314Th,0\n010403-20131011195414Mh,20131011195414Mh,20131011195414Mh,0\n010403-20131027025654Uh,20131027025654Uh,20131027025654Uh,2\n010403-20131107043054Uh,20131107043054Uh,20131107043054Uh,1\n010403-20140205180154Bh,20140205180154Bh,20140205180154Bh,2\n010403-20140216230134Lh,20140216230134Lh,20140216230134Lh,0\n010403-20140406195854Bh,20140406195854Bh,20140406195854Bh,0\n010403-20140413195914Mh,20140413195914Mh,20140413195914Mh,2\n010403-20140509071314Th,20140509071314Th,20140508195834Ch,1\n010403-20141006224634Lh,20141006224634Lh,20141006224634Lh,0\n010403-20141016195834Ch,20141016195834Ch,20141016022554Uh,1\n010403-20141106195854Bh,20141106195854Bh,20141105195814Mh,2\n010403-20141112195854Bh,20141112195854Bh,20141112195854Bh,0\n010403-20150125172714Mh,20150125172714Mh,20150125172714Mh,1\n010403-20150221155154Bh,20150221155154Bh,20150221155154Bh,2\n010403-20150228082414Th,20150228082414Th,20150228082414Th,0\n010403-20150308081954Uh,20150308081954Uh,20150308081954Uh,2\n010403-20150325015554Uh,20150325015554Uh,20150323172814Mh,2\n010403-20150423120034Ch,20150423120034Ch,20150423120034Ch,0\n010403-20150427162314Mh,20150427162314Mh,20150427162314Mh,1\n010403-20150614162514Mh,20150614162514Mh,20150614162514Mh,1\n010403-20150625082034Lh,20150625082034Lh,20150625082014Th,0\n010403-20150825082034Lh,20150825082034Lh,20150825082034Lh,1\n010403-20160213153054Bh,20160213153054Bh,20160213153054Bh,1\n010403-20160215133734Ch,20160215133734Ch,20160215133734Ch,0\n010403-20160225173314Mh,20160225173314Mh,20160225173314Mh,0\n010403-20160229151254Bh,20160229151254Bh,20160229151254Bh,1\n010403-20160403170114Mh,20160403170114Mh,20160401133734Ch,0\n010403-20160430133714Th,20160430133714Th,20160430133714Th,0\n010403-20160810234034Lh,20160810234034Lh,20160808133714Th,0\n010403-20160920230134Lh,20160920230134Lh,20160920230134Lh,2\n010403-20161205133714Th,20161205133714Th,20161205133714Th,0\n010403-20170119031254Uh,20170119031254Uh,20170119031254Uh,2\n010403-20170804140650Bh,20170804140650Bh,20170804140650Bh,1\n010403-20170927143450Bh,20170927143450Bh,20170927143450Bh,0\n010403-20171007133730Ch,20171007133730Ch,20171007133710Th,1\n010403-20171029150150Bh,20171029150150Bh,20171029150150Bh,0\n010403-20180306081810Th,20180306081810Th,20180306081810Th,2\n010403-20190412025430Lh,20190412025430Lh,20190412025430Lh,2\n010403-20201227202710Mh,20201227202710Mh,20201227202710Mh,1\n010403-20210223222830Ch,20210223222830Ch,20210222222850Bh,1\n010403-20210626060050Uh,20210626060050Uh,20210626060050Uh,1\n010403-20210817165030Mh,20210817165030Mh,20210817165030Mh,2\n010403-20220721185332Bh,20220721185332Bh,20220721185332Bh,1\n020101-20110406082634Lh,20110406082634Lh,20110406082634Lh,2\n020101-20110521063134Lh,20110521063134Lh,20110520063154Uh,0\n020101-20110526122534Ch,20110526122534Ch,20110525063154Uh,2\n020101-20110619063134Lh,20110619063134Lh,20110619063134Lh,0\n020101-20120330063154Uh,20120330063154Uh,20120330063134Lh,0\n020101-20120827085114Th,20120827085114Th,20120827085114Th,2\n020101-20121022145554Bh,20121022145554Bh,20121021145454Bh,1\n020101-20130303063134Lh,20130303063134Lh,20130303063134Lh,0\n020101-20130405114734Ch,20130405114734Ch,20130404063134Lh,2\n020101-20130609123334Ch,20130609123334Ch,20130609123334Ch,1\n020101-20130905073714Th,20130905073714Th,20130905073714Th,1\n020101-20131020080114Th,20131020080114Th,20131019080114Th,0\n020101-20131116185334Ch,20131116185334Ch,20131115185414Mh,0\n020101-20131204192154Bh,20131204192154Bh,20131204192154Bh,2\n020101-20140430071914Th,20140430071914Th,20140430071914Th,1\n020101-20140525070514Th,20140525070514Th,20140525070514Th,0\n020101-20140908175354Bh,20140908175354Bh,20140908175354Bh,2\n020101-20150330081954Uh,20150330081954Uh,20150330081954Uh,1\n020101-20150630083214Th,20150630083214Th,20150629161614Mh,2\n020101-20150820152254Bh,20150820152254Bh,20150819082014Th,0\n020101-20151106171514Mh,20151106171514Mh,20151105133734Ch,0\n020101-20151212171154Bh,20151212171154Bh,20151212171154Bh,1\n020101-20160201025254Uh,20160201025254Uh,20160201025254Uh,2\n020101-20160309172214Mh,20160309172214Mh,20160309172214Mh,2\n020101-20160518011654Uh,20160518011654Uh,20160518011654Uh,1\n020101-20161013133734Ch,20161013133734Ch,20161012133734Ch,2\n020101-20161109171714Mh,20161109171714Mh,20161108021454Uh,0\n020101-20170411013550Uh,20170411013550Uh,20170411013550Uh,2\n020101-20170701235630Lh,20170701235630Lh,20170701235630Lh,2\n020101-20170722235230Lh,20170722235230Lh,20170722235230Lh,2\n020101-20170801164710Mh,20170801164710Mh,20170801164710Mh,2\n020101-20180925000030Lh,20180925000030Lh,20180923112330Ch,0\n020101-20190927165910Mh,20190927165910Mh,20190925165810Mh,2\n020101-20200105073250Uh,20200105073250Uh,20200105073250Uh,0\n020101-20200619123010Th,20200619123010Th,20200619123010Th,0\n020101-20201022223330Lh,20201022223330Lh,20201022223330Lh,2\n020101-20210812044930Lh,20210812044930Lh,20210812044930Lh,0\n020101-20220718185352Mh,20220718185352Mh,20220718185352Mh,1\n020103-20150330081954Uh,20150330081954Uh,20150330081954Uh,1\n020201-20110520063154Uh,20110520063154Uh,20110520063154Uh,0\n020401-20110205082634Lh,20110205082634Lh,20110205082634Lh,2\n020401-20110324082614Th,20110324082614Th,20110324082614Th,2\n020401-20110510063134Lh,20110510063134Lh,20110509121334Ch,0\n020401-20110616133454Bh,20110616133454Bh,20110615133454Bh,1\n020401-20110707074514Th,20110707074514Th,20110707074514Th,2\n020401-20110815140454Bh,20110815140454Bh,20110815140454Bh,0\n020401-20111018063134Lh,20111018063134Lh,20111018063134Lh,2\n020401-20111112193254Bh,20111112193254Bh,20111112193254Bh,1\n020401-20120217181154Bh,20120217181154Bh,20120217063154Uh,1\n020401-20120223063154Uh,20120223063154Uh,20120223063154Uh,1\n020401-20120225172914Mh,20120225172914Mh,20120225172914Mh,0\n020401-20120318063154Uh,20120318063154Uh,20120318063154Uh,1\n020401-20120331142954Bh,20120331142954Bh,20120330063134Lh,0\n020401-20120403164114Mh,20120403164114Mh,20120403083114Th,2\n020401-20120814163114Mh,20120814163114Mh,20120813164714Mh,1\n020401-20121114172254Bh,20121114172254Bh,20121113103634Ch,0\n020401-20130110104734Ch,20130110104734Ch,20130109130334Ch,2\n020401-20130219063134Lh,20130219063134Lh,20130218220314Mh,0\n020401-20130314165714Mh,20130314165714Mh,20130314165714Mh,2\n020401-20130923060834Lh,20130923060834Lh,20130923060834Lh,2\n020401-20131101222734Lh,20131101222734Lh,20131030185354Bh,0\n020401-20131206084114Th,20131206084114Th,20131206084114Th,0\n020401-20140212024554Uh,20140212024554Uh,20140212024554Uh,1\n020401-20140806195834Ch,20140806195834Ch,20140806195834Ch,1\n020401-20141013195834Ch,20141013195834Ch,20141012015654Uh,1\n020401-20141105195814Mh,20141105195814Mh,20141105195814Mh,2\n020401-20141225024354Uh,20141225024354Uh,20141225024354Uh,1\n020401-20150610082014Th,20150610082014Th,20150610082014Th,2\n020401-20150617161314Mh,20150617161314Mh,20150617161314Mh,2\n020401-20150706083014Th,20150706083014Th,20150706083014Th,0\n020401-20150720082014Th,20150720082014Th,20150718082014Th,1\n020401-20160402025254Uh,20160402025254Uh,20160401133734Ch,0\n020401-20160501143154Bh,20160501143154Bh,20160430133714Th,0\n020401-20160628152654Bh,20160628152654Bh,20160627133734Ch,2\n020401-20170417014150Uh,20170417014150Uh,20170417014150Uh,1\n020401-20170425133730Ch,20170425133730Ch,20170425133730Ch,2\n020401-20171007133710Th,20171007133710Th,20171007133710Th,1\n020401-20171022170710Mh,20171022170710Mh,20171022170710Mh,1\n020401-20171025223130Lh,20171025223130Lh,20171025133730Ch,0\n020401-20171106171510Mh,20171106171510Mh,20171106171510Mh,2\n020401-20180318232230Lh,20180318232230Lh,20180318020350Uh,2\n020401-20200114073230Lh,20200114073230Lh,20200114073230Lh,0\n020401-20220122085210Th,20220122085210Th,20220122085210Th,1\n020403-20110205082634Lh,20110205082634Lh,20110205082634Lh,2\n020403-20110324082614Th,20110324082614Th,20110324082614Th,2\n020403-20110510063134Lh,20110510063134Lh,20110509121334Ch,0\n020403-20110616133454Bh,20110616133454Bh,20110615133454Bh,1\n020403-20110707074514Th,20110707074514Th,20110707074514Th,2\n020403-20110815140454Bh,20110815140454Bh,20110815140454Bh,0\n020403-20111018063134Lh,20111018063134Lh,20111018063134Lh,2\n020403-20111112193254Bh,20111112193254Bh,20111112193254Bh,1\n020403-20120217181154Bh,20120217181154Bh,20120217063154Uh,1\n020403-20120223063154Uh,20120223063154Uh,20120223063154Uh,1\n020403-20120225172914Mh,20120225172914Mh,20120225172914Mh,0\n020403-20120318063154Uh,20120318063154Uh,20120318063154Uh,1\n020403-20120331142954Bh,20120331142954Bh,20120330063134Lh,0\n020403-20120403164114Mh,20120403164114Mh,20120403083114Th,2\n020403-20120814163114Mh,20120814163114Mh,20120813164714Mh,1\n020403-20121114172254Bh,20121114172254Bh,20121113103634Ch,0\n020403-20130110104734Ch,20130110104734Ch,20130109130334Ch,2\n020403-20130219063134Lh,20130219063134Lh,20130218220314Mh,0\n020403-20130314165714Mh,20130314165714Mh,20130314165714Mh,2\n020403-20130923060834Lh,20130923060834Lh,20130923060834Lh,2\n020403-20131101222734Lh,20131101222734Lh,20131030185354Bh,0\n020403-20131206084114Th,20131206084114Th,20131206084114Th,0\n020403-20140212024554Uh,20140212024554Uh,20140212024554Uh,1\n020403-20140806195834Ch,20140806195834Ch,20140806195834Ch,1\n020403-20141013195834Ch,20141013195834Ch,20141012015654Uh,1\n020403-20141105195814Mh,20141105195814Mh,20141105195814Mh,2\n020403-20141225024354Uh,20141225024354Uh,20141225024354Uh,1\n020403-20150610082014Th,20150610082014Th,20150610082014Th,2\n020403-20150617161314Mh,20150617161314Mh,20150617161314Mh,2\n020403-20150706083014Th,20150706083014Th,20150706083014Th,0\n020403-20150720082014Th,20150720082014Th,20150718082014Th,1\n020403-20160402025254Uh,20160402025254Uh,20160401133734Ch,0\n020403-20160501143154Bh,20160501143154Bh,20160430133714Th,0\n020403-20160628152654Bh,20160628152654Bh,20160627133734Ch,2\n020403-20170417014150Uh,20170417014150Uh,20170417014150Uh,1\n020403-20170425133730Ch,20170425133730Ch,20170425133730Ch,2\n020403-20171022170710Mh,20171022170710Mh,20171022170710Mh,1\n020403-20171025223130Lh,20171025223130Lh,20171025133730Ch,0\n020403-20171106171510Mh,20171106171510Mh,20171106171510Mh,2\n020403-20180318232230Lh,20180318232230Lh,20180318020350Uh,2\n020403-20200114073230Lh,20200114073230Lh,20200114073230Lh,0\n020403-20220122085210Th,20220122085210Th,20220122085210Th,1\n030101-20110109104734Ch,20110109104734Ch,20110109104734Ch,2\n030101-20110317082654Uh,20110317082654Uh,20110317082654Uh,0\n030101-20110601063134Lh,20110601063134Lh,20110601063134Lh,1\n030101-20110617063134Lh,20110617063134Lh,20110615133454Bh,1\n030101-20110623063134Lh,20110623063134Lh,20110623063134Lh,2\n030101-20110902073614Th,20110902073614Th,20110902073614Th,1\n030101-20110923112234Ch,20110923112234Ch,20110923112234Ch,2\n030101-20120313063154Uh,20120313063154Uh,20120313063154Uh,2\n030101-20120323144054Bh,20120323144054Bh,20120323144054Bh,2\n030101-20120403083114Th,20120403083114Th,20120403083114Th,2\n030101-20121118092514Th,20121118092514Th,20121118092514Th,0\n030101-20130111104734Ch,20130111104734Ch,20130109130334Ch,2\n030101-20130201084914Th,20130201084914Th,20130201084914Th,0\n030101-20130225171214Mh,20130225171214Mh,20130225171214Mh,1\n030101-20130510134654Bh,20130510134654Bh,20130510134654Bh,1\n030101-20130615002434Lh,20130615002434Lh,20130615002434Lh,1\n030101-20130725122934Ch,20130725122934Ch,20130725122934Ch,0\n030101-20130805135854Bh,20130805135854Bh,20130805135854Bh,1\n030101-20130926060834Lh,20130926060834Lh,20130925143254Bh,1\n030101-20131031185614Mh,20131031185614Mh,20131030185354Bh,0\n030101-20140323020654Uh,20140323020654Uh,20140323020654Uh,2\n030101-20140719071414Th,20140719071414Th,20140719071414Th,2\n030101-20140720195834Ch,20140720195834Ch,20140719071414Th,2\n030101-20140816072814Th,20140816072814Th,20140816072814Th,1\n030101-20141013195854Bh,20141013195854Bh,20141012015654Uh,1\n030101-20141116222234Lh,20141116222234Lh,20141116222234Lh,1\n030101-20150130084914Th,20150130084914Th,20150130084914Th,2\n030101-20150203084714Th,20150203084714Th,20150203084714Th,1\n030101-20150501120634Ch,20150501120634Ch,20150501120634Ch,2\n030101-20150615133554Bh,20150615133554Bh,20150614162514Mh,1\n030101-20150719182314Mh,20150719182314Mh,20150718082014Th,1\n030101-20150919165814Mh,20150919165814Mh,20150919165814Mh,0\n030101-20151013144654Bh,20151013144654Bh,20151013144654Bh,1\n030101-20160103133734Ch,20160103133734Ch,20160103133734Ch,2\n030101-20160127225034Lh,20160127225034Lh,20160127225034Lh,1\n030101-20160317232434Lh,20160317232434Lh,20160317232434Lh,0\n030101-20160408232234Lh,20160408232234Lh,20160408232234Lh,1\n030101-20160411134234Ch,20160411134234Ch,20160411134234Ch,2\n030101-20160516012454Uh,20160516012454Uh,20160516012454Uh,1\n030101-20160609133654Bh,20160609133654Bh,20160609133654Bh,1\n030101-20160726135354Bh,20160726135354Bh,20160726135354Bh,1\n030101-20160831133714Th,20160831133714Th,20160831133714Th,2\n030101-20161116133714Th,20161116133714Th,20161116133714Th,0\n030101-20170427154550Bh,20170427154550Bh,20170427154550Bh,2\n030101-20171005133730Ch,20171005133730Ch,20171004143950Bh,2\n030101-20190925165810Mh,20190925165810Mh,20190925165810Mh,2\n030101-20210103223630Lh,20210103223630Lh,20210103223630Lh,1\n030101-20210815140550Bh,20210815140550Bh,20210814044930Lh,0\n030302-20110325171014Mh,20110325171014Mh,20110324082614Th,2\n030302-20110421164914Mh,20110421164914Mh,20110421164914Mh,0\n030302-20110530122834Ch,20110530122834Ch,20110530122834Ch,0\n030302-20110603163414Mh,20110603163414Mh,20110603163414Mh,2\n030302-20111107063154Uh,20111107063154Uh,20111107063154Uh,1\n030302-20120103104134Ch,20120103104134Ch,20120103104134Ch,1\n030302-20120227161654Bh,20120227161654Bh,20120227161654Bh,0\n030302-20120306063154Uh,20120306063154Uh,20120306063154Uh,0\n030302-20120424120134Ch,20120424120134Ch,20120424072414Th,1\n030302-20120712123634Ch,20120712123634Ch,20120712123634Ch,2\n030302-20120928063134Lh,20120928063134Lh,20120928063134Lh,1\n030302-20121026063134Lh,20121026063134Lh,20121026063134Lh,2\n030302-20130324015554Uh,20130324015554Uh,20130324015554Uh,2\n030302-20130609173414Mh,20130609173414Mh,20130609123334Ch,1\n030302-20131030185354Bh,20131030185354Bh,20131030185354Bh,0\n030302-20131109190514Mh,20131109190514Mh,20131107043054Uh,1\n030302-20131113185414Mh,20131113185414Mh,20131113004514Mh,1\n030302-20140501195814Mh,20140501195814Mh,20140430071914Th,1\n030302-20140523195814Mh,20140523195814Mh,20140522070614Th,1\n030302-20140615195814Mh,20140615195814Mh,20140615195814Mh,0\n030302-20140712195854Bh,20140712195854Bh,20140712195854Bh,1\n030302-20140725195814Mh,20140725195814Mh,20140725195814Mh,0\n030302-20140726071814Th,20140726071814Th,20140725195814Mh,0\n030302-20140921195854Bh,20140921195854Bh,20140921195854Bh,0\n030302-20141009023454Uh,20141009023454Uh,20141009023454Uh,2\n030302-20141025031154Uh,20141025031154Uh,20141024195834Ch,2\n030302-20150302082034Lh,20150302082034Lh,20150302082034Lh,2\n030302-20150625082014Th,20150625082014Th,20150625082014Th,0\n030302-20150718082014Th,20150718082014Th,20150718082014Th,1\n030302-20151011144554Bh,20151011144554Bh,20151011144554Bh,0\n030302-20151128222134Lh,20151128222134Lh,20151128133714Th,1\n030302-20160125154854Bh,20160125154854Bh,20160125154854Bh,2\n030302-20160510163914Mh,20160510163914Mh,20160510163914Mh,1\n030302-20160627133734Ch,20160627133734Ch,20160627133734Ch,2\n030302-20161108021454Uh,20161108021454Uh,20161108021454Uh,0\n030302-20161210133714Th,20161210133714Th,20161210133714Th,0\n030302-20170323041730Lh,20170323041730Lh,20170322015550Uh,0\n030302-20170408133710Th,20170408133710Th,20170408133710Th,2\n030302-20170417232530Lh,20170417232530Lh,20170417014150Uh,1\n030302-20170619012150Uh,20170619012150Uh,20170619012150Uh,1\n030302-20171025133730Ch,20171025133730Ch,20171025133730Ch,0\n030302-20171121222130Lh,20171121222130Lh,20171121222130Lh,0\n030302-20190629163710Mh,20190629163710Mh,20190629163710Mh,1\n030401-20111214132434Ch,20111214132434Ch,20111214063134Lh,1\n030401-20130407063154Uh,20130407063154Uh,20130407063154Uh,2\n030401-20140401014754Uh,20140401014754Uh,20140401014754Uh,1\n030401-20140922195854Bh,20140922195854Bh,20140921195854Bh,0\n030401-20150727082014Th,20150727082014Th,20150727082014Th,1\n030401-20151224133734Ch,20151224133734Ch,20151224133734Ch,1\n030401-20170925225630Lh,20170925225630Lh,20170925225630Lh,1\n030401-20211231223430Lh,20211231223430Lh,20211231223430Lh,2\n030403-20111115063134Lh,20111115063134Lh,20111114082314Th,2\n030403-20111214132434Ch,20111214132434Ch,20111214063134Lh,1\n030403-20120907141954Bh,20120907141954Bh,20120907141954Bh,2\n030403-20121108155954Bh,20121108155954Bh,20121108155954Bh,2\n030403-20130407063154Uh,20130407063154Uh,20130407063154Uh,2\n030403-20140401014754Uh,20140401014754Uh,20140401014754Uh,1\n030403-20140723071614Th,20140723071614Th,20140723071614Th,2\n030403-20150727082014Th,20150727082014Th,20150727082014Th,1\n030403-20170423014350Uh,20170423014350Uh,20170423014350Uh,0\n030403-20180506140230Ch,20180506140230Ch,20180506140230Ch,0\n030403-20210810121630Ch,20210810121630Ch,20210810121630Ch,1\n030403-20211231223430Lh,20211231223430Lh,20211231223430Lh,2\n040102-20110511063154Uh,20110511063154Uh,20110509121334Ch,0\n040102-20110525063154Uh,20110525063154Uh,20110525063154Uh,2\n040102-20110824141054Bh,20110824141054Bh,20110823120334Ch,1\n040102-20111214063134Lh,20111214063134Lh,20111214063134Lh,1\n040102-20120116063254Uh,20120116063254Uh,20120116063254Uh,2\n040102-20120427162614Mh,20120427162614Mh,20120427063154Uh,0\n040102-20120726071914Th,20120726071914Th,20120725122934Ch,1\n040102-20121207103034Ch,20121207103034Ch,20121207103034Ch,0\n040102-20130214111534Ch,20130214111534Ch,20130214111534Ch,2\n040102-20130326063134Lh,20130326063134Lh,20130326063134Lh,0\n040102-20130921060834Lh,20130921060834Lh,20130921060834Lh,2\n040102-20131201192134Ch,20131201192134Ch,20131130192134Ch,1\n040102-20140312025454Uh,20140312025454Uh,20140312025454Uh,0\n040102-20140603070414Th,20140603070414Th,20140603070414Th,0\n040102-20140607112354Uh,20140607112354Uh,20140607112354Uh,1\n040102-20140607195854Bh,20140607195854Bh,20140607112354Uh,1\n040102-20140802072114Th,20140802072114Th,20140802072114Th,1\n040102-20141007224934Lh,20141007224934Lh,20141006224634Lh,0\n040102-20141024195834Ch,20141024195834Ch,20141024195834Ch,2\n040102-20150119082034Lh,20150119082034Lh,20150119082034Lh,2\n040102-20150225171114Mh,20150225171114Mh,20150225171114Mh,0\n040102-20150315082034Lh,20150315082034Lh,20150314081954Uh,2\n040102-20151118172314Mh,20151118172314Mh,20151116201114Mh,1\n040102-20160129225334Lh,20160129225334Lh,20160129225334Lh,2\n040102-20160302133734Ch,20160302133734Ch,20160302133734Ch,0\n040102-20160604163414Mh,20160604163414Mh,20160604163414Mh,0\n040102-20161002225034Lh,20161002225034Lh,20161002225034Lh,2\n040102-20161201160954Bh,20161201160954Bh,20161201160954Bh,0\n040102-20161217133734Ch,20161217133734Ch,20161217133734Ch,2\n040102-20161223133734Ch,20161223133734Ch,20161222133714Th,1\n040102-20170702133730Ch,20170702133730Ch,20170701235630Lh,2\n040102-20171011224130Lh,20171011224130Lh,20171011170310Mh,1\n040102-20171109023950Uh,20171109023950Uh,20171109023950Uh,1\n040102-20171123133730Ch,20171123133730Ch,20171123133730Ch,0\n040102-20180710031950Uh,20180710031950Uh,20180710031950Uh,0\n040102-20200806053830Lh,20200806053830Lh,20200806053830Lh,1\n040102-20210731074710Th,20210731074710Th,20210731074710Th,1\n040102-20211229120130Ch,20211229120130Ch,20211229120130Ch,0\n040102-20220803074752Th,20220803074752Th,20220803074752Th,1\n040201-20111019063154Uh,20111019063154Uh,20111018063134Lh,2\n040201-20120725122934Ch,20120725122934Ch,20120725122934Ch,1\n040201-20130116172814Mh,20130116172814Mh,20130116172814Mh,2\n040201-20130306172214Mh,20130306172214Mh,20130303063134Lh,0\n040201-20130404063134Lh,20130404063134Lh,20130404063134Lh,2\n040201-20130611063154Uh,20130611063154Uh,20130611063154Uh,1\n040201-20130705161714Mh,20130705161714Mh,20130705161714Mh,2\n040201-20130716162214Mh,20130716162214Mh,20130716162214Mh,1\n040201-20140420085614Th,20140420085614Th,20140420085614Th,1\n040201-20140601195814Mh,20140601195814Mh,20140531195834Ch,1\n040201-20140821195834Ch,20140821195834Ch,20140821195834Ch,0\n040201-20140903073614Th,20140903073614Th,20140903073614Th,2\n040201-20160215175514Mh,20160215175514Mh,20160215133734Ch,0\n040201-20160808133714Th,20160808133714Th,20160808133714Th,0\n040201-20171014133730Ch,20171014133730Ch,20171014133730Ch,0\n040201-20171114133730Ch,20171114133730Ch,20171114133730Ch,2\n040201-20190704123830Ch,20190704123830Ch,20190704123830Ch,1\n040201-20210620163530Mh,20210620163530Mh,20210620163530Mh,2\n040201-20210728164530Mh,20210728164530Mh,20210728164530Mh,2\n040201-20210814044930Lh,20210814044930Lh,20210814044930Lh,0\n040301-20110711071214Th,20110711071214Th,20110711071214Th,1\n040301-20111201063154Uh,20111201063154Uh,20111201063154Uh,0\n040301-20111206102934Ch,20111206102934Ch,20111206102934Ch,2\n040301-20111208063134Lh,20111208063134Lh,20111208063134Lh,0\n040301-20120217063154Uh,20120217063154Uh,20120217063154Uh,1\n040301-20120218173614Mh,20120218173614Mh,20120217063154Uh,1\n040301-20120223152054Bh,20120223152054Bh,20120223063154Uh,1\n040301-20120328164614Mh,20120328164614Mh,20120328164614Mh,0\n040301-20120414175914Mh,20120414175914Mh,20120414115434Ch,0\n040301-20120430063154Uh,20120430063154Uh,20120430063154Uh,2\n040301-20120901114934Ch,20120901114934Ch,20120901114934Ch,2\n040301-20120914063134Lh,20120914063134Lh,20120913165314Mh,2\n040301-20130109130334Ch,20130109130334Ch,20130109130334Ch,2\n040301-20130211063134Lh,20130211063134Lh,20130211063134Lh,0\n040301-20130304170614Mh,20130304170614Mh,20130303063134Lh,0\n040301-20130502063134Lh,20130502063134Lh,20130502063134Lh,0\n040301-20130828141354Bh,20130828141354Bh,20130828141354Bh,2\n040301-20130913142354Bh,20130913142354Bh,20130913142354Bh,2\n040301-20131113004514Mh,20131113004514Mh,20131113004514Mh,1\n040301-20131130192134Ch,20131130192134Ch,20131130192134Ch,1\n040301-20140213024554Uh,20140213024554Uh,20140212024554Uh,1\n040301-20140218230234Lh,20140218230234Lh,20140218230234Lh,0\n040301-20140508195834Ch,20140508195834Ch,20140508195834Ch,1\n040301-20140609195854Bh,20140609195854Bh,20140609195854Bh,2\n040301-20140701195814Mh,20140701195814Mh,20140701195814Mh,1\n040301-20140807195854Bh,20140807195854Bh,20140806195834Ch,1\n040301-20140817195834Ch,20140817195834Ch,20140816072814Th,1\n040301-20140904195814Mh,20140904195814Mh,20140903073614Th,2\n040301-20150222090014Th,20150222090014Th,20150221155154Bh,2\n040301-20150222112034Ch,20150222112034Ch,20150221155154Bh,2\n040301-20151128133734Ch,20151128133734Ch,20151128133714Th,1\n040301-20151226133734Ch,20151226133734Ch,20151226133734Ch,2\n040301-20161012133734Ch,20161012133734Ch,20161012133734Ch,2\n040301-20170122034654Uh,20170122034654Uh,20170122034654Uh,0\n040301-20170612135150Bh,20170612135150Bh,20170612135150Bh,1\n040301-20180109223930Lh,20180109223930Lh,20180109223930Lh,0\n040301-20190701123830Ch,20190701123830Ch,20190701123830Ch,1\n040301-20210331231510Mh,20210331231510Mh,20210331231510Mh,1\n040301-20210626133950Bh,20210626133950Bh,20210626060050Uh,1\n040302-20110711071214Th,20110711071214Th,20110711071214Th,1\n040302-20120914063134Lh,20120914063134Lh,20120913165314Mh,2\n040302-20140508195834Ch,20140508195834Ch,20140508195834Ch,1\n040302-20140807195854Bh,20140807195854Bh,20140806195834Ch,1\n040302-20151128133734Ch,20151128133734Ch,20151128133714Th,1\n040302-20190701123830Ch,20190701123830Ch,20190701123830Ch,1\n040302-20210329015250Uh,20210329015250Uh,20210329015250Uh,0\n040302-20210331231510Mh,20210331231510Mh,20210331231510Mh,1\n040302-20220104043050Uh,20220104043050Uh,20220104043050Uh,0\n040303-20130828141354Bh,20130828141354Bh,20130828141354Bh,2\n040303-20140218230234Lh,20140218230234Lh,20140218230234Lh,0\n040303-20210626133950Bh,20210626133950Bh,20210626060050Uh,1\n040401-20110119082634Lh,20110119082634Lh,20110119082634Lh,0\n040401-20110312082634Lh,20110312082634Lh,20110312082634Lh,2\n040401-20110509121334Ch,20110509121334Ch,20110509121334Ch,0\n040401-20110728084714Th,20110728084714Th,20110727063234Lh,1\n040401-20111025080614Th,20111025080614Th,20111025080614Th,2\n040401-20120414115434Ch,20120414115434Ch,20120414115434Ch,0\n040401-20120421063154Uh,20120421063154Uh,20120421063154Uh,0\n040401-20120627162114Mh,20120627162114Mh,20120627162114Mh,1\n040401-20120803164614Mh,20120803164614Mh,20120802090314Th,0\n040401-20121018063134Lh,20121018063134Lh,20121018063134Lh,2\n040401-20121021145454Bh,20121021145454Bh,20121021145454Bh,1\n040401-20130824060834Lh,20130824060834Lh,20130824060834Lh,2\n040401-20130901073614Th,20130901073614Th,20130901073614Th,2\n040401-20131019080114Th,20131019080114Th,20131019080114Th,0\n040401-20131108045254Uh,20131108045254Uh,20131107043054Uh,1\n040401-20131115185414Mh,20131115185414Mh,20131115185414Mh,0\n040401-20140116180214Mh,20140116180214Mh,20140116180214Mh,0\n040401-20140522070614Th,20140522070614Th,20140522070614Th,1\n040401-20140709195814Mh,20140709195814Mh,20140709195814Mh,2\n040401-20140716195854Bh,20140716195854Bh,20140716195854Bh,1\n040401-20141010195834Ch,20141010195834Ch,20141009023454Uh,2\n040401-20150314081954Uh,20150314081954Uh,20150314081954Uh,2\n040401-20150324145954Bh,20150324145954Bh,20150323172814Mh,2\n040401-20150627133754Bh,20150627133754Bh,20150625082014Th,0\n040401-20150629161614Mh,20150629161614Mh,20150629161614Mh,2\n040401-20150830084134Lh,20150830084134Lh,20150830084134Lh,0\n040401-20150916055454Uh,20150916055454Uh,20150916055454Uh,1\n040401-20151107133734Ch,20151107133734Ch,20151105133734Ch,0\n040401-20151128133714Th,20151128133714Th,20151128133714Th,1\n040401-20151219133734Ch,20151219133734Ch,20151219133734Ch,0\n040401-20160327170714Mh,20160327170714Mh,20160326143654Bh,2\n040401-20160619133714Th,20160619133714Th,20160619133714Th,0\n040401-20160629133714Th,20160629133714Th,20160627133734Ch,2\n040401-20160927133734Ch,20160927133734Ch,20160927133734Ch,0\n040401-20170310173450Bh,20170310173450Bh,20170310133730Ch,1\n040401-20170805133710Th,20170805133710Th,20170804140650Bh,1\n040401-20170806133710Th,20170806133710Th,20170804140650Bh,1\n040401-20171115222230Lh,20171115222230Lh,20171114133730Ch,2\n040401-20180712123630Ch,20180712123630Ch,20180712123630Ch,0\n040401-20180714000010Mh,20180714000010Mh,20180714000010Mh,1\n040401-20210616142650Bh,20210616142650Bh,20210616142650Bh,2\n040401-20211119103130Ch,20211119103130Ch,20211119103130Ch,1\n040401-20220618185332Bh,20220618185332Bh,20220617185352Mh,2\n040402-20171115222230Lh,20171115222230Lh,20171114133730Ch,2\n040402-20190805054730Lh,20190805054730Lh,20190805054730Lh,1\n040402-20210616142650Bh,20210616142650Bh,20210616142650Bh,2\n050101-20110322005814Mh,20110322005814Mh,20110322005814Mh,1\n050101-20110429135554Bh,20110429135554Bh,20110429135554Bh,2\n050101-20110502063154Uh,20110502063154Uh,20110502063154Uh,0\n050101-20110517164514Mh,20110517164514Mh,20110517164514Mh,2\n050101-20110727063234Lh,20110727063234Lh,20110727063234Lh,1\n050101-20110826074214Th,20110826074214Th,20110826074214Th,1\n050101-20111114082314Th,20111114082314Th,20111114082314Th,2\n050101-20111116063134Lh,20111116063134Lh,20111114082314Th,2\n050101-20111218085014Th,20111218085014Th,20111218085014Th,2\n050101-20120110155754Bh,20120110155754Bh,20120110155754Bh,2\n050101-20120715123434Ch,20120715123434Ch,20120715123434Ch,0\n050101-20120720085114Th,20120720085114Th,20120720085114Th,1\n050101-20120810121634Ch,20120810121634Ch,20120810121634Ch,0\n050101-20120811121434Ch,20120811121434Ch,20120810121634Ch,0\n050101-20120816072814Th,20120816072814Th,20120816072814Th,0\n050101-20121007131134Ch,20121007131134Ch,20121006063534Lh,0\n050101-20121027165114Mh,20121027165114Mh,20121026063134Lh,2\n050101-20130107174854Bh,20130107174854Bh,20130107174854Bh,2\n050101-20130218220314Mh,20130218220314Mh,20130218220314Mh,0\n050101-20130305112734Ch,20130305112734Ch,20130303063134Lh,0\n050101-20130630070714Th,20130630070714Th,20130630070714Th,2\n050101-20130719140154Bh,20130719140154Bh,20130719140154Bh,0\n050101-20131225065154Uh,20131225065154Uh,20131224192154Bh,0\n050101-20131229112414Th,20131229112414Th,20131229112414Th,0\n050101-20140210180154Bh,20140210180154Bh,20140210180154Bh,2\n050101-20140414195834Ch,20140414195834Ch,20140413195914Mh,2\n050101-20140710083214Th,20140710083214Th,20140709195814Mh,2\n050101-20141214201834Ch,20141214201834Ch,20141214201834Ch,0\n050101-20141216222634Lh,20141216222634Lh,20141216222634Lh,0\n050101-20150113105034Ch,20150113105034Ch,20150112081954Uh,0\n050101-20150116082034Lh,20150116082034Lh,20150116082034Lh,1\n050101-20150216171714Mh,20150216171714Mh,20150216171714Mh,2\n050101-20150421140454Bh,20150421140454Bh,20150421140454Bh,0\n050101-20150731082034Lh,20150731082034Lh,20150731082034Lh,0\n050101-20160101133714Th,20160101133714Th,20160101133714Th,1\n050101-20160310133734Ch,20160310133734Ch,20160309172214Mh,2\n050101-20160617133654Bh,20160617133654Bh,20160617133654Bh,1\n050101-20170117175014Mh,20170117175014Mh,20170117175014Mh,0\n050101-20170609133710Th,20170609133710Th,20170609133710Th,1\n050101-20170728150230Ch,20170728150230Ch,20170728150230Ch,1\n050101-20170827165310Mh,20170827165310Mh,20170827165310Mh,0\n050101-20171115170950Bh,20171115170950Bh,20171114133730Ch,2\n050101-20171123222130Lh,20171123222130Lh,20171123133730Ch,0\n050101-20180202130330Ch,20180202130330Ch,20180202130330Ch,2\n050101-20181101230010Mh,20181101230010Mh,20181101230010Mh,0\n050101-20220328143550Bh,20220328143550Bh,20220328143550Bh,0\n050101-20220622205632Bh,20220622205632Bh,20220622205632Bh,2\n050102-20110322005814Mh,20110322005814Mh,20110322005814Mh,1\n050102-20110429135554Bh,20110429135554Bh,20110429135554Bh,2\n050102-20110502063154Uh,20110502063154Uh,20110502063154Uh,0\n050102-20110517164514Mh,20110517164514Mh,20110517164514Mh,2\n050102-20110727063234Lh,20110727063234Lh,20110727063234Lh,1\n050102-20110826074214Th,20110826074214Th,20110826074214Th,1\n050102-20111114082314Th,20111114082314Th,20111114082314Th,2\n050102-20111116063134Lh,20111116063134Lh,20111114082314Th,2\n050102-20111218085014Th,20111218085014Th,20111218085014Th,2\n050102-20120110155754Bh,20120110155754Bh,20120110155754Bh,2\n050102-20120715123434Ch,20120715123434Ch,20120715123434Ch,0\n050102-20120720085114Th,20120720085114Th,20120720085114Th,1\n050102-20120811121434Ch,20120811121434Ch,20120810121634Ch,0\n050102-20120816072814Th,20120816072814Th,20120816072814Th,0\n050102-20121027165114Mh,20121027165114Mh,20121026063134Lh,2\n050102-20130107174854Bh,20130107174854Bh,20130107174854Bh,2\n050102-20130218220314Mh,20130218220314Mh,20130218220314Mh,0\n050102-20130305112734Ch,20130305112734Ch,20130303063134Lh,0\n050102-20130630070714Th,20130630070714Th,20130630070714Th,2\n050102-20130719140154Bh,20130719140154Bh,20130719140154Bh,0\n050102-20131225065154Uh,20131225065154Uh,20131224192154Bh,0\n050102-20131229112414Th,20131229112414Th,20131229112414Th,0\n050102-20140210180154Bh,20140210180154Bh,20140210180154Bh,2\n050102-20140414195834Ch,20140414195834Ch,20140413195914Mh,2\n050102-20140710083214Th,20140710083214Th,20140709195814Mh,2\n050102-20141214201834Ch,20141214201834Ch,20141214201834Ch,0\n050102-20141216222634Lh,20141216222634Lh,20141216222634Lh,0\n050102-20150113105034Ch,20150113105034Ch,20150112081954Uh,0\n050102-20150116082034Lh,20150116082034Lh,20150116082034Lh,1\n050102-20150216171714Mh,20150216171714Mh,20150216171714Mh,2\n050102-20150421140454Bh,20150421140454Bh,20150421140454Bh,0\n050102-20150731082034Lh,20150731082034Lh,20150731082034Lh,0\n050102-20160101133714Th,20160101133714Th,20160101133714Th,1\n050102-20160310133734Ch,20160310133734Ch,20160309172214Mh,2\n050102-20160617133654Bh,20160617133654Bh,20160617133654Bh,1\n050102-20170117175014Mh,20170117175014Mh,20170117175014Mh,0\n050102-20170609133710Th,20170609133710Th,20170609133710Th,1\n050102-20170728150230Ch,20170728150230Ch,20170728150230Ch,1\n050102-20170827165310Mh,20170827165310Mh,20170827165310Mh,0\n050102-20171123222130Lh,20171123222130Lh,20171123133730Ch,0\n050102-20180202130330Ch,20180202130330Ch,20180202130330Ch,2\n050102-20220328143550Bh,20220328143550Bh,20220328143550Bh,0\n050102-20220622205632Bh,20220622205632Bh,20220622205632Bh,2\n050103-20110322005814Mh,20110322005814Mh,20110322005814Mh,1\n050103-20110429135554Bh,20110429135554Bh,20110429135554Bh,2\n050103-20110502063154Uh,20110502063154Uh,20110502063154Uh,0\n050103-20110517164514Mh,20110517164514Mh,20110517164514Mh,2\n050103-20110727063234Lh,20110727063234Lh,20110727063234Lh,1\n050103-20110826074214Th,20110826074214Th,20110826074214Th,1\n050103-20111114082314Th,20111114082314Th,20111114082314Th,2\n050103-20111116063134Lh,20111116063134Lh,20111114082314Th,2\n050103-20111218085014Th,20111218085014Th,20111218085014Th,2\n050103-20120110155754Bh,20120110155754Bh,20120110155754Bh,2\n050103-20120715123434Ch,20120715123434Ch,20120715123434Ch,0\n050103-20120720085114Th,20120720085114Th,20120720085114Th,1\n050103-20120810121634Ch,20120810121634Ch,20120810121634Ch,0\n050103-20120811121434Ch,20120811121434Ch,20120810121634Ch,0\n050103-20120816072814Th,20120816072814Th,20120816072814Th,0\n050103-20121007131134Ch,20121007131134Ch,20121006063534Lh,0\n050103-20121027165114Mh,20121027165114Mh,20121026063134Lh,2\n050103-20130107174854Bh,20130107174854Bh,20130107174854Bh,2\n050103-20130218220314Mh,20130218220314Mh,20130218220314Mh,0\n050103-20130305112734Ch,20130305112734Ch,20130303063134Lh,0\n050103-20130630070714Th,20130630070714Th,20130630070714Th,2\n050103-20130719140154Bh,20130719140154Bh,20130719140154Bh,0\n050103-20131225065154Uh,20131225065154Uh,20131224192154Bh,0\n050103-20131229112414Th,20131229112414Th,20131229112414Th,0\n050103-20140210180154Bh,20140210180154Bh,20140210180154Bh,2\n050103-20140414195834Ch,20140414195834Ch,20140413195914Mh,2\n050103-20140710083214Th,20140710083214Th,20140709195814Mh,2\n050103-20141214201834Ch,20141214201834Ch,20141214201834Ch,0\n050103-20141216222634Lh,20141216222634Lh,20141216222634Lh,0\n050103-20150113105034Ch,20150113105034Ch,20150112081954Uh,0\n050103-20150116082034Lh,20150116082034Lh,20150116082034Lh,1\n050103-20150216171714Mh,20150216171714Mh,20150216171714Mh,2\n050103-20150421140454Bh,20150421140454Bh,20150421140454Bh,0\n050103-20150731082034Lh,20150731082034Lh,20150731082034Lh,0\n050103-20160101133714Th,20160101133714Th,20160101133714Th,1\n050103-20160310133734Ch,20160310133734Ch,20160309172214Mh,2\n050103-20160617133654Bh,20160617133654Bh,20160617133654Bh,1\n050103-20170117175014Mh,20170117175014Mh,20170117175014Mh,0\n050103-20170609133710Th,20170609133710Th,20170609133710Th,1\n050103-20170728150230Ch,20170728150230Ch,20170728150230Ch,1\n050103-20170827165310Mh,20170827165310Mh,20170827165310Mh,0\n050103-20171115170950Bh,20171115170950Bh,20171114133730Ch,2\n050103-20171123222130Lh,20171123222130Lh,20171123133730Ch,0\n050103-20180202130330Ch,20180202130330Ch,20180202130330Ch,2\n050103-20180801232450Bh,20180801232450Bh,20180801232450Bh,0\n050103-20181101230010Mh,20181101230010Mh,20181101230010Mh,0\n050103-20220328143550Bh,20220328143550Bh,20220328143550Bh,0\n050103-20220622205632Bh,20220622205632Bh,20220622205632Bh,2\n050201-20160928143554Bh,20160928143554Bh,20160927133734Ch,0\n050202-20110425063154Uh,20110425063154Uh,20110425063154Uh,0\n050202-20120127085214Th,20120127085214Th,20120127085214Th,0\n050202-20120213063134Lh,20120213063134Lh,20120213063134Lh,0\n050202-20120523163014Mh,20120523163014Mh,20120523163014Mh,0\n050202-20120730135454Bh,20120730135454Bh,20120729122634Ch,2\n050202-20121219063154Uh,20121219063154Uh,20121218063534Lh,2\n050202-20130616070414Th,20130616070414Th,20130615002434Lh,1\n050202-20131013185334Ch,20131013185334Ch,20131013185334Ch,1\n050202-20140128174554Bh,20140128174554Bh,20140127180134Ch,2\n050202-20140803234534Lh,20140803234534Lh,20140802072114Th,1\n050202-20140908231334Lh,20140908231334Lh,20140908175354Bh,2\n050202-20150327145114Th,20150327145114Th,20150327145114Th,2\n050202-20150621161314Mh,20150621161314Mh,20150620082014Th,2\n050202-20150913082014Th,20150913082014Th,20150913082014Th,1\n050202-20150926111934Ch,20150926111934Ch,20150926111934Ch,2\n050202-20151116224034Lh,20151116224034Lh,20151116201114Mh,1\n050202-20160326143654Bh,20160326143654Bh,20160326143654Bh,2\n050202-20160430133734Ch,20160430133734Ch,20160430133714Th,0\n050202-20160720133734Ch,20160720133734Ch,20160720133734Ch,1\n050202-20160928143554Bh,20160928143554Bh,20160927133734Ch,0\n050202-20161229133734Ch,20161229133734Ch,20161229133734Ch,2\n050202-20170322015550Uh,20170322015550Uh,20170322015550Uh,0\n050202-20171005144050Bh,20171005144050Bh,20171004143950Bh,2\n050302-20110211084114Th,20110211084114Th,20110211084114Th,0\n050302-20120113104834Ch,20120113104834Ch,20120113104834Ch,0\n050302-20130808140054Bh,20130808140054Bh,20130807073114Th,2\n050302-20170102024454Uh,20170102024454Uh,20170102024454Uh,0\n050302-20170503013050Uh,20170503013050Uh,20170502233130Lh,1\n050302-20181102104630Ch,20181102104630Ch,20181101230010Mh,0\n050302-20190317145150Bh,20190317145150Bh,20190317145150Bh,0\n050302-20190926165810Mh,20190926165810Mh,20190925165810Mh,2\n050303-20110810063134Lh,20110810063134Lh,20110810063134Lh,2\n050303-20120113104834Ch,20120113104834Ch,20120113104834Ch,0\n050303-20121208171514Mh,20121208171514Mh,20121207103034Ch,0\n050303-20141001195834Ch,20141001195834Ch,20141001195834Ch,1\n050303-20150305081954Uh,20150305081954Uh,20150305081954Uh,1\n050303-20150620082014Th,20150620082014Th,20150620082014Th,2\n050303-20151105133734Ch,20151105133734Ch,20151105133734Ch,0\n050303-20151209201754Bh,20151209201754Bh,20151209201754Bh,0\n050303-20160925225634Lh,20160925225634Lh,20160925165814Mh,1\n050303-20190926165810Mh,20190926165810Mh,20190925165810Mh,2\n060102-20110326170914Mh,20110326170914Mh,20110324082614Th,2\n060102-20130222082914Th,20130222082914Th,20130221063134Lh,2\n060102-20130419070254Uh,20130419070254Uh,20130419070254Uh,1\n060102-20130423072414Th,20130423072414Th,20130423072414Th,0\n060102-20140122180214Mh,20140122180214Mh,20140122180214Mh,2\n060102-20140818195814Mh,20140818195814Mh,20140816072814Th,1\n060102-20150207084414Th,20150207084414Th,20150207084414Th,1\n060102-20150323172814Mh,20150323172814Mh,20150323172814Mh,2\n060102-20151019223534Lh,20151019223534Lh,20151019223534Lh,0\n060102-20151228024254Uh,20151228024254Uh,20151228024254Uh,2\n060102-20160724135554Bh,20160724135554Bh,20160724135554Bh,0\n060102-20160809164914Mh,20160809164914Mh,20160808133714Th,0\n060102-20161023133814Th,20161023133814Th,20161023133814Th,2\n060102-20170217165250Bh,20170217165250Bh,20170217165250Bh,0\n060102-20170316133730Ch,20170316133730Ch,20170316133730Ch,1\n060102-20170809164910Mh,20170809164910Mh,20170809164910Mh,1\n060102-20180827232530Lh,20180827232530Lh,20180827232530Lh,1\n060102-20180924112030Ch,20180924112030Ch,20180923112330Ch,0\n060102-20190322144450Bh,20190322144450Bh,20190322144450Bh,0\n060102-20201228202430Ch,20201228202430Ch,20201227202710Mh,1\n060102-20210329114430Ch,20210329114430Ch,20210329015250Uh,0\n060102-20210721123430Ch,20210721123430Ch,20210721123430Ch,1\n060102-20220724185352Mh,20220724185352Mh,20220724185352Mh,2\n060403-20110123174914Mh,20110123174914Mh,20110123174914Mh,2\n060403-20110221173814Mh,20110221173814Mh,20110220082634Lh,2\n060403-20110313082654Uh,20110313082654Uh,20110312082634Lh,2\n060403-20110523063154Uh,20110523063154Uh,20110523063154Uh,2\n060403-20110615133454Bh,20110615133454Bh,20110615133454Bh,1\n060403-20110621063134Lh,20110621063134Lh,20110621063134Lh,2\n060403-20111226155454Bh,20111226155454Bh,20111226155454Bh,0\n060403-20120114063154Uh,20120114063154Uh,20120113104834Ch,0\n060403-20120509163414Mh,20120509163414Mh,20120509163414Mh,1\n060403-20120813164714Mh,20120813164714Mh,20120813164714Mh,1\n060403-20120814140454Bh,20120814140454Bh,20120813164714Mh,1\n060403-20120829063134Lh,20120829063134Lh,20120829063134Lh,0\n060403-20121030150354Bh,20121030150354Bh,20121030150354Bh,1\n060403-20130212153254Bh,20130212153254Bh,20130211063134Lh,0\n060403-20130327175114Mh,20130327175114Mh,20130326063134Lh,0\n060403-20130411115134Ch,20130411115134Ch,20130411115134Ch,1\n060403-20130601063154Uh,20130601063154Uh,20130601063154Uh,1\n060403-20130628063134Lh,20130628063134Lh,20130628063134Lh,1\n060403-20130707134154Bh,20130707134154Bh,20130707134154Bh,2\n060403-20131028190734Ch,20131028190734Ch,20131027025654Uh,2\n060403-20131207084214Th,20131207084214Th,20131206084114Th,0\n060403-20140411013854Uh,20140411013854Uh,20140411013854Uh,2\n060403-20140705035754Uh,20140705035754Uh,20140705035754Uh,0\n060403-20140814195854Bh,20140814195854Bh,20140813064254Uh,2\n060403-20140927014554Uh,20140927014554Uh,20140927014554Uh,1\n060403-20141012195854Bh,20141012195854Bh,20141012015654Uh,1\n060403-20141026024454Uh,20141026024454Uh,20141024195834Ch,2\n060403-20141225172214Mh,20141225172214Mh,20141225024354Uh,1\n060403-20150113155654Bh,20150113155654Bh,20150112081954Uh,0\n060403-20150409081954Uh,20150409081954Uh,20150409081954Uh,0\n060403-20150417082314Th,20150417082314Th,20150417082314Th,1\n060403-20150527082014Th,20150527082014Th,20150527082014Th,2\n060403-20151116201114Mh,20151116201114Mh,20151116201114Mh,1\n060403-20160107133714Th,20160107133714Th,20160107133714Th,2\n060403-20160110133714Th,20160110133714Th,20160110133714Th,0\n060403-20160530234734Lh,20160530234734Lh,20160530234734Lh,1\n060403-20160804133734Ch,20160804133734Ch,20160804133734Ch,0\n060403-20160925165814Mh,20160925165814Mh,20160925165814Mh,1\n060403-20170125133734Ch,20170125133734Ch,20170125133734Ch,0\n060403-20180301050430Lh,20180301050430Lh,20180301050430Lh,2\n060403-20180923112330Ch,20180923112330Ch,20180923112330Ch,0\n060403-20181014105830Ch,20181014105830Ch,20181014105830Ch,0\n060403-20211222120130Ch,20211222120130Ch,20211222120130Ch,1\n'}  # filled by code/build_e006_kernel.py: {filename: source text}
EMBEDDED_FOLDS_SHA256 = '6316d1e5cd628f01b0073ab12caea91c38533a4e31366e2a1360a6863115b6ec'
SOURCE_REF = 'nomannic19/cv-solar-filament-mask-r-cnn-u-net-refiner'
SOURCE_SNAPSHOT_SHA256 = 'c6b004aa5dca9edde71ba73972d0c69e9e97173430abac1ab1af8661b93fed6f'  # filled by the builder (snapshots/notebooks/.../v0.ipynb)

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


class DetDataset:
    """Source FilamentDataset: one (selected) annotator record per physical image."""

    def __init__(self, items, segs, size, augment):
        self.items, self.segs, self.size, self.augment = items, segs, size, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        import cv2
        import torch
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
    torch.backends.cudnn.benchmark = True
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
    train_ds = DetDataset(cfg['train_items'], segs, size, augment=True)
    sampler = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=SEED)
    nw = cfg['workers']
    train_loader = DataLoader(train_ds, batch_size=1, sampler=sampler, num_workers=nw, pin_memory=use_cuda,
                              persistent_workers=nw > 0, collate_fn=collate_fn, worker_init_fn=seed_worker)
    val_loader = None
    if rank == 0:
        val_ds = DetDataset(cfg['val_items'], segs, size, augment=False)
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
            metrics = source_val(core, val_loader, device, size)
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
    ap.add_argument('--image-size', type=int, default=1536, help='source 1024; smaller only for CPU smoke')
    ap.add_argument('--seed', type=int, default=2027, help='source 2026; B0 seed rerun uses another')
    ap.add_argument('--train-records', default='selected', choices=['selected', 'random', 'union'],
                    help='detector training labels: source = record with most filaments; random = any record per sample (B1)')
    ap.add_argument('--input-mode', default='phys', choices=['gray', 'phys'], help='detector input channels (E034 phys)')
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
    ap.add_argument('--cap-seconds', type=float, default=18000.0, help='server session cap (3 h)')
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
    val_ids = [p for p in pids if phys_fold[p] == args.fold]
    tr_ids = [p for p in pids if phys_fold[p] != args.fold]
    if args.max_val_images:
        val_ids = val_ids[:args.max_val_images]
    if not val_ids or not tr_ids or set(phys_fold[p] for p in tr_ids) & {args.fold}:
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
           'input_mode': args.input_mode,
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
    va_img = np.concatenate([x[0] for x in prepped[ntr:]]); va_msk = np.concatenate([x[1] for x in prepped[ntr:]])
    entries = {p: prepped[ntr + j][3] for j, p in enumerate(val_ids)}
    receipt['timing']['crop_cache_seconds'] = time.time() - tp
    receipt['refiner_data'] = {'train_crops': len(tr_img), 'val_crops': len(va_img),
                               'empty_annotation_masks_skipped': sum(x[2] for x in prepped),
                               'val_entries': sum(len(v) for v in entries.values())}
    del prepped
    log(f'crop cache {receipt["refiner_data"]} in {receipt["timing"]["crop_cache_seconds"]:.1f}s')

    device = torch.device('cuda:0' if dev_type == 'cuda' else 'cpu')  # first CUDA use in this process (after fork pool)
    receipt['device']['cuda_devices'] = [torch.cuda.get_device_name(i) for i in range(n_gpu)]
    torch.backends.cudnn.benchmark = True
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
