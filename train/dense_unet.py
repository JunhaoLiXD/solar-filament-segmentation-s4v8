"""E056 (D1): copy of the E004/E022 U-Net script with only the encoder made configurable (default ConvNeXt-T, timm
fb_in22k_ft_in1k, Apache-2.0) and run locally at 1536. Output file names keep the e004_ prefix so existing loaders work.
Local: python code/e056_dense_unet.py --input data --output runs/e056-convnext-fold0 --folds-csv data/folds.csv --fold 0
       --size 1536 --skip-pip --no-wheels --cap-seconds 30000

Original docstring (E004): smp U-Net (resnet34 ImageNet encoder, 1-channel) fold-0 baseline.

plan.md section 2 + decisions.md 2026-10-27 "Accept all six Codex Critic findings".
Order: preprocess -> train folds != 0 -> save checkpoint -> save fold-0 OOF probs (fp16 @1024)
-> receipt -> fresh-process reload + held-out forward -> RLE -> validator round-trip
-> fixed-default PQ -> time-bounded PQ grid (tuned PQ is selection-biased).

Kaggle: python e004_unet.py   (writes to /kaggle/working)
Local smoke: python e004_unet.py --input <fake> --output <dir> --size 256 --epochs 2 --batch 2
             --encoder-weights none --skip-pip --device cpu --folds-csv <csv>
"""
import argparse
import csv
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

T0 = time.monotonic()
EMBEDDED = {'pq_metric.py': '"""NumPy reproduction of official self-evaluation notebook cells 3-4.\n\nSource: snapshots/official-self-evaluation/self-evaluation-notebook.ipynb.\nThis establishes local source semantics only, not online-service equivalence.\nRepeated annotator entries reuse physical-image predictions. All qualifying\npairs count, with no one-to-one assignment. Extra-only physical images are\nignored by the source entry construction; zero-GT entries cannot originate\nfrom an annotation-row table. Do not exploit either limitation.\n"""\nimport numpy as np\n\n\ndef overlap_matrix(gt, pred):\n    gt, pred = np.asarray(gt), np.asarray(pred)\n    if gt.ndim != 3 or pred.ndim != 3 or gt.shape[1:] != pred.shape[1:]:\n        raise ValueError(\'Expected matching (instances,height,width) arrays\')\n    if not np.isin(gt, [0, 1]).all() or not np.isin(pred, [0, 1]).all():\n        raise ValueError(\'Masks must be binary\')\n    # Official arrays are float32; preserve that arithmetic rather than uint8\n    # matmul (which silently overflows) or a differently rounded float64 path.\n    g = gt.astype(np.float32).reshape(len(gt), -1) if len(gt) else np.empty((0, np.prod(gt.shape[1:])), np.float32)\n    p = pred.astype(np.float32).reshape(len(pred), -1) if len(pred) else np.empty((0, np.prod(pred.shape[1:])), np.float32)\n    intersection = g @ p.T\n    union = g.sum(axis=1)[:, None] + p.sum(axis=1)[None, :] - intersection\n    return np.divide(intersection, union, out=np.zeros_like(intersection), where=union != 0)\n\n\ndef score_matrices(matrices):\n    """Globally aggregate overlap matrices; preserve explicit empty-entry branch."""\n    tp, fp, fn, tp_values = 0, 0, 0, []\n    for matrix in matrices:\n        matrix = np.asarray(matrix)\n        if matrix.ndim != 2 or not np.isfinite(matrix).all() or ((matrix < 0) | (matrix > 1)).any():\n            raise ValueError(\'Invalid IoU matrix\')\n        n_gt, n_pred = matrix.shape\n        if n_gt == 0:\n            fp += n_pred\n            continue\n        if n_pred == 0:\n            fn += n_gt\n            continue\n        hit = matrix > .5\n        values = matrix[hit].tolist()\n        tp += len(values)\n        tp_values.extend(values)\n        fp += int((hit.sum(axis=0) == 0).sum())\n        fn += int((hit.sum(axis=1) == 0).sum())\n    denominator = tp + .5 * fp + .5 * fn\n    iou_sum = sum(tp_values)\n    return {\'pq\': iou_sum / denominator if denominator else 0.0, \'tp\': tp, \'fp\': fp, \'fn\': fn, \'tp_iou_sum\': iou_sum}\n\n\ndef evaluate_entries(entries, predictions):\n    """entries: iterable (annotator-image ID, physical ID, GT stack).\n\n    Caller supplies one entry per annotator-image represented in annotation\n    rows, matching official get_overlap_df construction. Repeated physical IDs\n    intentionally reuse predictions. Missing predictions become empty stacks.\n    """\n    matrices, seen, used = [], set(), set()\n    for entry_id, physical_id, gt in entries:\n        if entry_id in seen:\n            raise ValueError(\'Duplicate annotator-image entry\')\n        seen.add(entry_id)\n        gt = np.asarray(gt)\n        if gt.ndim != 3:\n            raise ValueError(\'Invalid GT shape\')\n        pred = predictions.get(physical_id, np.zeros((0, *gt.shape[1:]), dtype=np.uint8))\n        matrices.append(overlap_matrix(gt, pred))\n        used.add(physical_id)\n    result = score_matrices(matrices)\n    result.update(entries=len(seen), ignored_extra_prediction_images=sorted(set(predictions) - used),\n                  scope=\'official local source cells 3-4; online scorer parity unknown\')\n    return result\n', 'gpu_smoke_audit.py': '"""Original offline Kaggle audit + tiny training smoke test; never submits or infers test."""\nimport argparse\nimport csv\nimport hashlib\nimport importlib.metadata\nimport json\nimport math\nimport os\nfrom pathlib import Path\nimport re\nimport threading\nimport time\nimport traceback\n\nSTART = time.monotonic()\n\n\ndef dump(path, value):\n    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")\n\n\ndef terminal_receipt(out, status, details):\n    path = out / "audit.json"\n    try:\n        audit = json.loads(path.read_text(encoding="utf-8"))\n    except (OSError, ValueError):\n        audit = {}\n    audit.update(status=status, baseline_ready=False, full_audit_complete=False,\n                 total_seconds=time.monotonic() - START, terminal_details=details)\n    dump(path, audit)\n\n\ndef write_csv(path, rows):\n    if not rows:\n        raise ValueError(f"No rows for {path}")\n    with path.open("w", newline="", encoding="utf-8") as stream:\n        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))\n        writer.writeheader()\n        writer.writerows(rows)\n\n\ndef parse_name(path):\n    match = re.fullmatch(r"(\\d{8})(\\d{6})([A-Za-z]{2})", path.stem)\n    if not match:\n        raise ValueError(f"Unexpected physical image name: {path.name}")\n    from datetime import datetime\n    datetime.strptime(match[1] + match[2], "%Y%m%d%H%M%S")\n    return match[1], match[3]\n\n\ndef main(args):\n    # Import heavy dependencies only after the hard wall-clock watchdog is active.\n    import numpy as np\n    import PIL\n    from PIL import Image\n    import torch\n    from torch import nn\n    from pycocotools import mask as coco_mask\n\n    out = args.output\n    effective_config = {"steps": args.steps, "wall_seconds": args.timeout,\n                        "input": str(args.input), "output": str(args.output),\n                        "requested_accelerator": "NvidiaTeslaT4", "server_timeout_expected_seconds": 450,\n                        "maximum_allocated_devices": 2, "maximum_aggregate_gpu_hours": 0.25,\n                        "architecture": "original Conv1-8-8-1", "seed": 2026,\n                        "image_resize": [256, 256], "maximum_batch": 4}\n    audit = {"purpose": "environment and partial data audit only; no baseline CV", "status": "running",\n             "baseline_ready": False, "full_audit_complete": False,\n             "stages": {"structural_audit": "running", "mask_decoding": "pending", "training": "pending", "checkpoint": "pending"},\n             "outstanding_checks": ["full mask decoding and foreground distributions", "category/bbox/area/license integrity", "near-duplicate audit", "final balanced folds", "official metric parity", "held-out inference and checkpoint reload/export"],\n             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),\n             "effective_config": effective_config,\n             "config_sha256": hashlib.sha256(json.dumps(effective_config, sort_keys=True).encode()).hexdigest(),\n             "limits": {"steps": args.steps, "wall_seconds": args.timeout},\n             "versions": {"torch": torch.__version__, "numpy": np.__version__, "Pillow": PIL.__version__, "pycocotools": importlib.metadata.version("pycocotools")}}\n    dump(out / "audit.json", audit)\n    if not torch.cuda.is_available():\n        raise RuntimeError("CUDA unavailable: refusing CPU/local training fallback")\n    audit["cuda"] = {"version": torch.version.cuda, "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}\n    dump(out / "audit.json", audit)\n    if torch.cuda.device_count() not in (1, 2):\n        raise RuntimeError("This T4 diagnostic requires one or two allocated CUDA devices")\n    candidates = []\n    for path in args.input.rglob("*.json"):\n        if "train" in path.name.lower() and "annotation" in path.name.lower():\n            candidates.append(path)\n    if len(candidates) != 1:\n        raise ValueError(f"Expected exactly one train annotation JSON; found {candidates}")\n    annotation_path = candidates[0]\n    data = json.loads(annotation_path.read_text(encoding="utf-8"))\n    if not isinstance(data, dict) or not isinstance(data.get("images"), list) or not isinstance(data.get("annotations"), list):\n        raise ValueError("Annotation JSON must contain COCO images and annotations lists")\n    if not data["images"] or not data["annotations"]:\n        raise ValueError("Empty training images or annotations")\n    files = {}\n    split_for = {}\n    for path in args.input.rglob("*"):\n        if path.suffix.lower() not in {".jpg", ".jpeg"}:\n            continue\n        dirs = {p.lower() for p in path.parts}\n        split = "train" if "train_images" in dirs else "test" if "test_images" in dirs else None\n        if split is None:\n            continue\n        if path.name in files or path.stem in split_for:\n            raise ValueError(f"Ambiguous duplicate mounted image name: {path}")\n        files[path.name] = path\n        split_for[path.stem] = split\n    if set(split_for.values()) != {"train", "test"}:\n        raise ValueError("Expected mounted train_images and test_images JPEG directories")\n    physical = {}\n    for name, path in sorted(files.items()):\n        day, station = parse_name(path)\n        with Image.open(path) as im:\n            if im.size != (2048, 2048):\n                raise ValueError(f"Invalid image dimensions: {path}: {im.size}")\n            if im.mode not in ("L", "RGB"):\n                raise ValueError(f"Unexpected JPEG mode {im.mode}: {path}")\n            if im.mode == "RGB":\n                rgb = np.asarray(im)\n                if not (np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])):\n                    raise ValueError(f"Non-grayscale RGB image: {path}")\n            pixels = np.asarray(im.convert("L"))\n            digest = hashlib.sha256(pixels.tobytes()).hexdigest()\n        physical[path.stem] = {"physical_id": path.stem, "split": split_for[path.stem], "day": day,\n                               "station": station, "pixel_sha256": digest, "path": str(path)}\n    records = {}\n    counts = {}\n    annotations = {}\n    for row in data["images"]:\n        key = str(row["id"])\n        name = Path(row["file_name"]).name\n        if key in records or name not in files or split_for.get(Path(name).stem) != "train":\n            raise ValueError(f"Duplicate image record ID or unresolved train image: {key}, {name}")\n        if (row.get("width"), row.get("height")) != (2048, 2048):\n            raise ValueError(f"Bad COCO dimensions: {key}")\n        records[key] = row\n        counts[key] = 0\n        annotations[key] = []\n    seen_annotations = set()\n    segmentation_types = {}\n    for row in data["annotations"]:\n        key, aid = str(row["image_id"]), str(row["id"])\n        if key not in records or aid in seen_annotations:\n            raise ValueError(f"Orphan or duplicate annotation: {aid}, image {key}")\n        seen_annotations.add(aid)\n        seg = row.get("segmentation")\n        if isinstance(seg, list) and seg:\n            for polygon in seg:\n                if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:\n                    raise ValueError(f"Invalid polygon structure: {aid}")\n                if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 2048 for v in polygon):\n                    raise ValueError(f"Nonfinite/out-of-bounds polygon: {aid}")\n            kind = "polygon"\n        elif isinstance(seg, dict) and seg.get("size") == [2048, 2048] and isinstance(seg.get("counts"), (list, str)):\n            kind = "rle"\n            # Validate every native RLE, rather than passing malformed buffers to C.\n            counts_value = seg["counts"]\n            if isinstance(counts_value, str):\n                counts_value = parse_compressed_counts(counts_value)\n            if not counts_value or any(type(v) is not int or v < 0 for v in counts_value) or sum(counts_value) != 2048 * 2048:\n                raise ValueError(f"Invalid RLE counts: {aid}")\n        else:\n            raise ValueError(f"Unsupported or empty segmentation: {aid}")\n        segmentation_types[kind] = segmentation_types.get(kind, 0) + 1\n        annotations[key].append(row)\n        counts[key] += 1\n    unreferenced = sorted(set(p for p, v in physical.items() if v["split"] == "train") - {Path(r["file_name"]).stem for r in records.values()})\n    if unreferenced:\n        raise ValueError(f"Train files absent from annotations: {unreferenced[:10]}")\n    # Candidate connected components join same-day observations and exact pixel duplicates.\n    parent = {p: p for p in physical}\n    def find(x):\n        while parent[x] != x:\n            parent[x] = parent[parent[x]]\n            x = parent[x]\n        return x\n    def union(a, b):\n        a, b = find(a), find(b)\n        parent[max(a, b)] = min(a, b)\n    days, hashes = {}, {}\n    for p, row in physical.items():\n        for value, groups in ((row["day"], days), (row["pixel_sha256"], hashes)):\n            if value in groups:\n                union(p, groups[value][0])\n            groups.setdefault(value, []).append(p)\n    fold_rows, manifest = [], []\n    for key, row in sorted(records.items()):\n        p = Path(row["file_name"]).stem\n        group = find(p)\n        fold = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5\n        fold_rows.append({"image_id": key, "physical_id": p, "group": group, "fold": fold})\n        manifest.append({"image_id": key, **physical[p], "annotation_count": counts[key], "license": row.get("license")})\n    write_csv(out / "manifest.csv", manifest)\n    write_csv(out / "folds_candidate.csv", fold_rows)\n    dump(out / "group_report.json", {"policy": "candidate only: connected same-day / exact pixel duplicate groups; not final CV",\n         "exact_duplicate_groups": [v for v in hashes.values() if len(v) > 1],\n         "multi_image_day_groups": {k: v for k, v in days.items() if len(v) > 1},\n         "physical_images": list(physical.values()),\n         "train_test_day_overlap": [k for k, v in days.items() if len({physical[p]["split"] for p in v}) > 1]})\n    audit.update({"annotation_file": str(annotation_path), "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),\n                  "physical_train": sum(v["split"] == "train" for v in physical.values()),\n                  "physical_test": sum(v["split"] == "test" for v in physical.values()),\n                  "annotator_image_records": len(records), "annotation_count": len(seen_annotations),\n                  "segmentation_types": segmentation_types, "licenses": data.get("licenses"),\n                  "fold_record_counts": {i: sum(r["fold"] == i for r in fold_rows) for i in range(5)},\n                  "audit_seconds": time.monotonic() - START})\n    dump(out / "audit.json", audit)\n    # One annotator\'s union mask per chosen physical image; this is only a plumbing test.\n    xs, ys, selected, chosen = [], [], [], set()\n    decoded_annotation_records = 0\n    for key, row in sorted(records.items()):\n        p = Path(row["file_name"]).stem\n        if p in chosen or counts[key] == 0:\n            continue\n        chosen.add(p)\n        mask = np.zeros((2048, 2048), dtype=np.uint8)\n        for ann in annotations[key]:\n            seg = ann["segmentation"]\n            if isinstance(seg, list):\n                rle = coco_mask.merge(coco_mask.frPyObjects(seg, 2048, 2048))\n            elif isinstance(seg["counts"], list):\n                rle = coco_mask.frPyObjects(seg, 2048, 2048)\n            else:\n                rle = {"size": [2048, 2048], "counts": seg["counts"].encode("ascii")}\n            decoded = coco_mask.decode(rle)\n            if decoded.shape != mask.shape:\n                raise ValueError(f"Decoded shape mismatch: {ann[\'id\']}")\n            mask |= decoded\n            decoded_annotation_records += 1\n        with Image.open(files[Path(row["file_name"]).name]) as im:\n            xs.append(np.asarray(im.convert("L").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255)\n        ys.append(np.asarray(Image.fromarray(mask).resize((256, 256), Image.Resampling.NEAREST), dtype=np.float32))\n        selected.append(key)\n        if len(xs) == 4:\n            break\n    if len(xs) < 2 or not any(np.any(y) for y in ys):\n        raise ValueError("Need at least two real annotated images and nonempty targets")\n    audit.update(decoded_annotation_records=decoded_annotation_records,\n                 decoded_annotator_image_records=len(selected))\n    audit["stages"].update(structural_audit="partial", mask_decoding="selected_records_only", training="running")\n    dump(out / "audit.json", audit)\n    torch.manual_seed(2026)\n    torch.cuda.manual_seed_all(2026)\n    model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1),\n                          nn.ReLU(), nn.Conv2d(8, 1, 1)).cuda()\n    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)\n    x = torch.from_numpy(np.stack(xs)[:, None]).cuda()\n    y = torch.from_numpy(np.stack(ys)[:, None]).cuda()\n    training_start = time.monotonic()\n    losses, step_seconds = [], []\n    for step in range(args.steps):\n        if time.monotonic() - START > args.timeout - 30:\n            raise TimeoutError("Stopped before hard timeout; insufficient training time")\n        tick = time.monotonic()\n        optimizer.zero_grad(set_to_none=True)\n        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)\n        if not torch.isfinite(loss):\n            raise ValueError("Nonfinite training loss")\n        loss.backward()\n        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):\n            raise ValueError("Nonfinite training gradients")\n        optimizer.step()\n        torch.cuda.synchronize()\n        losses.append(float(loss.detach().cpu()))\n        step_seconds.append(time.monotonic() - tick)\n    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},\n                "architecture": "original Conv1-8-8-1", "smoke_only": True, "training_image_ids": selected}, out / "tiny_checkpoint.pt")\n    audit["stages"].update(training="success", checkpoint="saved_not_reload_validated")\n    audit.update({"status": "environment_smoke_success", "training_ids": selected, "training_steps": len(losses), "losses": losses,\n                  "step_seconds": step_seconds, "train_seconds": time.monotonic() - training_start,\n                  "total_seconds": time.monotonic() - START, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),\n                  "test_inference_seconds": 0, "cv_score": None})\n    dump(out / "audit.json", audit)\n    print(json.dumps(audit), flush=True)\n\n\ndef parse_compressed_counts(value):\n    """Bounded Python COCO counts parser, before any native decoder."""\n    counts, pos = [], 0\n    while pos < len(value):\n        number, shift = 0, 0\n        while True:\n            if pos >= len(value) or shift > 30:\n                raise ValueError("Truncated or oversized compressed RLE")\n            char = ord(value[pos]) - 48\n            pos += 1\n            if not 0 <= char <= 63:\n                raise ValueError("Invalid compressed RLE character")\n            number |= (char & 31) << shift\n            shift += 5\n            if not char & 32:\n                if char & 16:\n                    number |= -1 << shift\n                break\n        if len(counts) > 2:\n            number += counts[-2]\n        if number < 0 or number > 2048 * 2048:\n            raise ValueError("RLE run out of bounds")\n        counts.append(number)\n        if len(counts) > 2048 * 2048 + 1:\n            raise ValueError("Too many RLE runs")\n    return counts\n\n\nif __name__ == "__main__":\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))\n    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))\n    parser.add_argument("--steps", type=int, default=10)\n    parser.add_argument("--timeout", type=int, default=360)\n    args = parser.parse_args()\n    if not 1 <= args.steps <= 20 or not 60 <= args.timeout <= 360:\n        parser.error("steps must be 1..20; timeout 60..360 (server limit must be 450)")\n    if not Path("/kaggle").is_dir() or args.input.resolve() != Path("/kaggle/input").resolve():\n        parser.error("This smoke test is Kaggle-only and must use /kaggle/input")\n    args.output.mkdir(parents=True, exist_ok=True)\n    def hard_stop():\n        details = {"status": "hard_timeout", "elapsed_seconds": time.monotonic() - START}\n        terminal_receipt(args.output, "hard_timeout", details)\n        dump(args.output / "timeout.json", details)\n        os._exit(124)\n    timer = threading.Timer(max(1, args.timeout - (time.monotonic() - START)), hard_stop)\n    timer.daemon = True\n    timer.start()\n    try:\n        main(args)\n    except Exception:\n        details = {"status": "failed", "elapsed_seconds": time.monotonic() - START,\n                   "traceback": traceback.format_exc()}\n        terminal_receipt(args.output, "failed", details)\n        dump(args.output / "failure.json", details)\n        raise\n    finally:\n        timer.cancel()\n', 'validate_submission.py': '"""Validate variable-row filament instance CSV against the actual test manifest."""\nimport argparse\nimport csv\nimport json\nfrom pathlib import Path, PurePosixPath\n\nfrom gpu_smoke_audit import parse_compressed_counts\n\n# Valid high-frequency 2048x2048 masks can exceed Python CSV\'s 128 KiB default.\nCSV_FIELD_LIMIT = 64 * 1024 * 1024\n\n\ndef test_ids_from_manifest(path):\n    manifest = json.loads(Path(path).read_text(encoding="utf-8"))\n    if manifest.get("complete") is not True:\n        raise ValueError("A complete official file manifest is required")\n    ids = []\n    for row in manifest["files"]:\n        file = PurePosixPath(row["name"])\n        if "test_images" in file.parts and file.suffix.lower() in {".jpg", ".jpeg"}:\n            ids.append(file.stem)\n    if not ids or len(ids) != len(set(ids)):\n        raise ValueError("Manifest test IDs are empty or ambiguous")\n    return set(ids)\n\n\ndef validate_rle(value):\n    from pycocotools import mask as coco_mask\n    if not value or value != value.strip() or value.startswith(("b\'", \'b"\', "{")):\n        raise ValueError("RLE must be the compressed counts string alone")\n    counts = parse_compressed_counts(value)\n    if not counts or sum(counts) != 2048 * 2048:\n        raise ValueError("RLE runs must cover exactly 2048 x 2048 pixels")\n    # The bounded parser above rejects malformed sizes before entering native code.\n    decoded = coco_mask.decode({"size": [2048, 2048], "counts": value.encode("ascii")})\n    if decoded.shape != (2048, 2048):\n        raise ValueError("RLE decoded dimensions do not match [2048, 2048]")\n    # Canonical roundtrip prevents accepting trailing garbage / alternate byte syntax.\n    canonical = coco_mask.encode(decoded)["counts"].decode("ascii")\n    if canonical != value:\n        raise ValueError("RLE is not canonical compressed COCO counts")\n    return int(decoded.sum())\n\n\ndef validate_submission(path, manifest=None, *, allowed_ids=None, reject_empty_masks=False, reject_all_empty=False):\n    csv.field_size_limit(max(csv.field_size_limit(), CSV_FIELD_LIMIT))\n    if allowed_ids is None:\n        if manifest is None:\n            manifest = Path(__file__).resolve().parents[1] / "research/data-file-manifest.json"\n        test_ids = test_ids_from_manifest(manifest)\n    else:\n        values = list(allowed_ids)\n        if not values or len(values) != len(set(values)) or any(not isinstance(v, str) or not v or v != v.strip() or \'_\' in v or any(c.isspace() for c in v) for v in values):\n            raise ValueError("Explicit allowed held-out IDs must be nonempty, unique physical-image stems")\n        test_ids = set(values)\n    seen, predicted, empty_masks, total_area = set(), set(), 0, 0\n    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:\n        reader = csv.DictReader(stream, strict=True)\n        if reader.fieldnames != ["filament_id", "segmentation_rle"]:\n            raise ValueError("Expected exactly filament_id,segmentation_rle in that order")\n        for line, row in enumerate(reader, 2):\n            try:\n                if None in row or None in row.values():\n                    raise ValueError("Wrong field count")\n                instance = row["filament_id"]\n                if instance in seen:\n                    raise ValueError(f"Duplicate instance ID: {instance}")\n                if instance != instance.strip() or "_" not in instance:\n                    raise ValueError("Instance ID needs original image stem and nonempty underscore suffix")\n                image, suffix = instance.split("_", 1)\n                if image not in test_ids or not suffix or any(c.isspace() for c in suffix):\n                    raise ValueError(f"Foreign image ID or invalid instance suffix: {instance}")\n                area = validate_rle(row["segmentation_rle"])\n                if reject_empty_masks and area == 0:\n                    raise ValueError("Candidate policy rejects empty instance masks")\n                empty_masks += area == 0\n                total_area += area\n                seen.add(instance)\n                predicted.add(image)\n            except (ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:\n                raise ValueError(f"CSV line {line}: {exc}") from exc\n    if reject_all_empty and total_area == 0:\n        raise ValueError("Candidate policy rejects all-empty output (including no rows)")\n    missing = sorted(test_ids - predicted)\n    warnings = []\n    if missing:\n        warnings.append(f"{len(missing)} test images have no predicted rows; server coverage/empty-image conventions remain unverified")\n    if empty_masks:\n        warnings.append(f"{empty_masks} empty instance masks; server acceptance remains unverified")\n    return {"valid": True, "rows": len(seen), "predicted_images": len(predicted),\n            "test_images": len(test_ids), "missing_image_ids": missing,\n            "empty_masks": empty_masks, "warnings": warnings,\n            "id_scope": "explicit held-out IDs" if allowed_ids is not None else "official test manifest",\n            "candidate_policy": {"reject_empty_masks": reject_empty_masks, "reject_all_empty": reject_all_empty},\n            "scope": "format validation only; no exact row-count, full coverage or server-score guarantee"}\n\n\nif __name__ == "__main__":\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("csv", type=Path)\n    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "research/data-file-manifest.json")\n    parser.add_argument("--allowed-heldout-ids", type=Path, help="JSON list of permitted physical image stems; replaces test manifest ID scope")\n    parser.add_argument("--reject-empty-masks", action="store_true")\n    parser.add_argument("--reject-all-empty", action="store_true")\n    args = parser.parse_args()\n    try:\n        ids = json.loads(args.allowed_heldout_ids.read_text(encoding="utf-8")) if args.allowed_heldout_ids else None\n        if ids is not None and not isinstance(ids, list):\n            raise ValueError("Allowed held-out IDs file must contain a JSON list")\n        print(json.dumps(validate_submission(args.csv, args.manifest, allowed_ids=ids,\n                                             reject_empty_masks=args.reject_empty_masks, reject_all_empty=args.reject_all_empty), indent=2))\n    except (ValueError, csv.Error, OSError) as exc:\n        parser.exit(1, f"Validation failed: {exc}\\n")\n', 'folds.csv': 'record_id,physical_id,group_id,fold\n010101-20110114105034Ch,20110114105034Ch,20110114105034Ch,1\n010101-20110203082634Lh,20110203082634Lh,20110203082634Lh,2\n010101-20110220082634Lh,20110220082634Lh,20110220082634Lh,2\n010101-20110327143554Bh,20110327143554Bh,20110324082614Th,2\n010101-20110625163814Mh,20110625163814Mh,20110625163814Mh,2\n010101-20110914063134Lh,20110914063134Lh,20110914063134Lh,1\n010101-20110918112834Ch,20110918112834Ch,20110918112834Ch,1\n010101-20111027145954Bh,20111027145954Bh,20111027145954Bh,2\n010101-20111029063154Uh,20111029063154Uh,20111029063154Uh,0\n010101-20120206063134Lh,20120206063134Lh,20120205172414Mh,2\n010101-20120215153654Bh,20120215153654Bh,20120215153654Bh,0\n010101-20120330063134Lh,20120330063134Lh,20120330063134Lh,0\n010101-20120427063154Uh,20120427063154Uh,20120427063154Uh,0\n010101-20120603123034Ch,20120603123034Ch,20120603123034Ch,2\n010101-20121109103634Ch,20121109103634Ch,20121108155954Bh,2\n010101-20130425135954Bh,20130425135954Bh,20130425135954Bh,2\n010101-20130811140254Bh,20130811140254Bh,20130807073114Th,2\n010101-20130903060834Lh,20130903060834Lh,20130903060834Lh,0\n010101-20131006075314Th,20131006075314Th,20131006075314Th,0\n010101-20131016185334Ch,20131016185334Ch,20131016185334Ch,1\n010101-20131017185334Ch,20131017185334Ch,20131016185334Ch,1\n010101-20131025185414Mh,20131025185414Mh,20131025185414Mh,0\n010101-20131109222434Lh,20131109222434Lh,20131107043054Uh,1\n010101-20131128192214Mh,20131128192214Mh,20131128192214Mh,1\n010101-20140127180134Ch,20140127180134Ch,20140127180134Ch,2\n010101-20140312081114Th,20140312081114Th,20140312025454Uh,0\n010101-20140321195814Mh,20140321195814Mh,20140321195814Mh,1\n010101-20140509195814Mh,20140509195814Mh,20140508195834Ch,1\n010101-20140513012554Uh,20140513012554Uh,20140513012554Uh,2\n010101-20140604195854Bh,20140604195854Bh,20140603070414Th,0\n010101-20140813064254Uh,20140813064254Uh,20140813064254Uh,2\n010101-20141027025054Uh,20141027025054Uh,20141024195834Ch,2\n010101-20141027224834Lh,20141027224834Lh,20141024195834Ch,2\n010101-20141129222134Lh,20141129222134Lh,20141129222134Lh,1\n010101-20141210201814Mh,20141210201814Mh,20141210201814Mh,0\n010101-20150112081954Uh,20150112081954Uh,20150112081954Uh,0\n010101-20150518145654Bh,20150518145654Bh,20150518145654Bh,1\n010101-20150527161214Mh,20150527161214Mh,20150527082014Th,2\n010101-20150531082034Lh,20150531082034Lh,20150531082034Lh,1\n010101-20150626161514Mh,20150626161514Mh,20150625082014Th,0\n010101-20150806082034Lh,20150806082034Lh,20150806082034Lh,0\n010101-20150927111834Ch,20150927111834Ch,20150926111934Ch,2\n010101-20151117172314Mh,20151117172314Mh,20151116201114Mh,1\n010101-20151122172614Mh,20151122172614Mh,20151122172614Mh,2\n010101-20160226173214Mh,20160226173214Mh,20160225173314Mh,0\n010101-20160320133714Th,20160320133714Th,20160320133714Th,0\n010101-20160401170314Mh,20160401170314Mh,20160401133734Ch,0\n010101-20160526011254Uh,20160526011254Uh,20160526011254Uh,2\n010101-20160606163514Mh,20160606163514Mh,20160606163514Mh,2\n010101-20160703133714Th,20160703133714Th,20160703133714Th,0\n010101-20160714235434Lh,20160714235434Lh,20160714235434Lh,1\n010101-20161129022454Uh,20161129022454Uh,20161129022454Uh,0\n010101-20161222133714Th,20161222133714Th,20161222133714Th,1\n010101-20161224024154Uh,20161224024154Uh,20161222133714Th,1\n010101-20170103030554Uh,20170103030554Uh,20170102024454Uh,0\n010101-20170107133714Th,20170107133714Th,20170106133714Th,2\n010101-20170218173710Mh,20170218173710Mh,20170217165250Bh,0\n010101-20170418013450Uh,20170418013450Uh,20170417014150Uh,1\n010101-20171008144250Bh,20171008144250Bh,20171007133710Th,1\n010101-20171011170310Mh,20171011170310Mh,20171011170310Mh,1\n010101-20171027223030Lh,20171027223030Lh,20171027223030Lh,0\n010101-20171101133730Ch,20171101133730Ch,20171101133730Ch,2\n010101-20180309000910Mh,20180309000910Mh,20180309000910Mh,0\n010101-20180318020350Uh,20180318020350Uh,20180318020350Uh,2\n010101-20180825205030Ch,20180825205030Ch,20180825205030Ch,0\n010101-20181023232430Lh,20181023232430Lh,20181023232430Lh,1\n010101-20190602025430Lh,20190602025430Lh,20190602025430Lh,1\n010101-20190626025430Lh,20190626025430Lh,20190626025430Lh,2\n010101-20200110073230Lh,20200110073230Lh,20200110073230Lh,1\n010101-20200116073250Uh,20200116073250Uh,20200116073250Uh,2\n010101-20210222222850Bh,20210222222850Bh,20210222222850Bh,1\n010101-20211016145050Bh,20211016145050Bh,20211016145050Bh,2\n010102-20110203082634Lh,20110203082634Lh,20110203082634Lh,2\n010102-20110220082634Lh,20110220082634Lh,20110220082634Lh,2\n010102-20110327143554Bh,20110327143554Bh,20110324082614Th,2\n010102-20110914063134Lh,20110914063134Lh,20110914063134Lh,1\n010102-20110918112834Ch,20110918112834Ch,20110918112834Ch,1\n010102-20111027145954Bh,20111027145954Bh,20111027145954Bh,2\n010102-20111029063154Uh,20111029063154Uh,20111029063154Uh,0\n010102-20120206063134Lh,20120206063134Lh,20120205172414Mh,2\n010102-20120215153654Bh,20120215153654Bh,20120215153654Bh,0\n010102-20120330063134Lh,20120330063134Lh,20120330063134Lh,0\n010102-20120427063154Uh,20120427063154Uh,20120427063154Uh,0\n010102-20120603123034Ch,20120603123034Ch,20120603123034Ch,2\n010102-20121109103634Ch,20121109103634Ch,20121108155954Bh,2\n010102-20130425135954Bh,20130425135954Bh,20130425135954Bh,2\n010102-20130811140254Bh,20130811140254Bh,20130807073114Th,2\n010102-20130903060834Lh,20130903060834Lh,20130903060834Lh,0\n010102-20131006075314Th,20131006075314Th,20131006075314Th,0\n010102-20131016185334Ch,20131016185334Ch,20131016185334Ch,1\n010102-20131017185334Ch,20131017185334Ch,20131016185334Ch,1\n010102-20131025185414Mh,20131025185414Mh,20131025185414Mh,0\n010102-20131109222434Lh,20131109222434Lh,20131107043054Uh,1\n010102-20131128192214Mh,20131128192214Mh,20131128192214Mh,1\n010102-20140112180134Ch,20140112180134Ch,20140111224134Lh,1\n010102-20140127180134Ch,20140127180134Ch,20140127180134Ch,2\n010102-20140312081114Th,20140312081114Th,20140312025454Uh,0\n010102-20140321195814Mh,20140321195814Mh,20140321195814Mh,1\n010102-20140509195814Mh,20140509195814Mh,20140508195834Ch,1\n010102-20140513012554Uh,20140513012554Uh,20140513012554Uh,2\n010102-20140604195854Bh,20140604195854Bh,20140603070414Th,0\n010102-20140813064254Uh,20140813064254Uh,20140813064254Uh,2\n010102-20141027025054Uh,20141027025054Uh,20141024195834Ch,2\n010102-20141027224834Lh,20141027224834Lh,20141024195834Ch,2\n010102-20141129222134Lh,20141129222134Lh,20141129222134Lh,1\n010102-20141210201814Mh,20141210201814Mh,20141210201814Mh,0\n010102-20150112081954Uh,20150112081954Uh,20150112081954Uh,0\n010102-20150518145654Bh,20150518145654Bh,20150518145654Bh,1\n010102-20150527161214Mh,20150527161214Mh,20150527082014Th,2\n010102-20150531082034Lh,20150531082034Lh,20150531082034Lh,1\n010102-20150626161514Mh,20150626161514Mh,20150625082014Th,0\n010102-20150806082034Lh,20150806082034Lh,20150806082034Lh,0\n010102-20150927111834Ch,20150927111834Ch,20150926111934Ch,2\n010102-20151117172314Mh,20151117172314Mh,20151116201114Mh,1\n010102-20151122172614Mh,20151122172614Mh,20151122172614Mh,2\n010102-20160226173214Mh,20160226173214Mh,20160225173314Mh,0\n010102-20160320133714Th,20160320133714Th,20160320133714Th,0\n010102-20160401170314Mh,20160401170314Mh,20160401133734Ch,0\n010102-20160526011254Uh,20160526011254Uh,20160526011254Uh,2\n010102-20160606163514Mh,20160606163514Mh,20160606163514Mh,2\n010102-20160703133714Th,20160703133714Th,20160703133714Th,0\n010102-20160714235434Lh,20160714235434Lh,20160714235434Lh,1\n010102-20161129022454Uh,20161129022454Uh,20161129022454Uh,0\n010102-20161222133714Th,20161222133714Th,20161222133714Th,1\n010102-20161224024154Uh,20161224024154Uh,20161222133714Th,1\n010102-20170103030554Uh,20170103030554Uh,20170102024454Uh,0\n010102-20170218173710Mh,20170218173710Mh,20170217165250Bh,0\n010102-20170418013450Uh,20170418013450Uh,20170417014150Uh,1\n010102-20171008144250Bh,20171008144250Bh,20171007133710Th,1\n010102-20171011170310Mh,20171011170310Mh,20171011170310Mh,1\n010102-20171027223030Lh,20171027223030Lh,20171027223030Lh,0\n010102-20171101133730Ch,20171101133730Ch,20171101133730Ch,2\n010102-20180318020350Uh,20180318020350Uh,20180318020350Uh,2\n010102-20180825205030Ch,20180825205030Ch,20180825205030Ch,0\n010102-20190602025430Lh,20190602025430Lh,20190602025430Lh,1\n010102-20190626025430Lh,20190626025430Lh,20190626025430Lh,2\n010102-20200110073230Lh,20200110073230Lh,20200110073230Lh,1\n010103-20110114105034Ch,20110114105034Ch,20110114105034Ch,1\n010103-20110203082634Lh,20110203082634Lh,20110203082634Lh,2\n010103-20110914063134Lh,20110914063134Lh,20110914063134Lh,1\n010103-20111027145954Bh,20111027145954Bh,20111027145954Bh,2\n010103-20120603123034Ch,20120603123034Ch,20120603123034Ch,2\n010103-20121109103634Ch,20121109103634Ch,20121108155954Bh,2\n010103-20131016185334Ch,20131016185334Ch,20131016185334Ch,1\n010103-20140112180134Ch,20140112180134Ch,20140111224134Lh,1\n010103-20140509195814Mh,20140509195814Mh,20140508195834Ch,1\n010103-20140604195854Bh,20140604195854Bh,20140603070414Th,0\n010103-20150112081954Uh,20150112081954Uh,20150112081954Uh,0\n010103-20150927111834Ch,20150927111834Ch,20150926111934Ch,2\n010103-20151117172314Mh,20151117172314Mh,20151116201114Mh,1\n010103-20151122172614Mh,20151122172614Mh,20151122172614Mh,2\n010103-20160320133714Th,20160320133714Th,20160320133714Th,0\n010103-20160401170314Mh,20160401170314Mh,20160401133734Ch,0\n010103-20160526011254Uh,20160526011254Uh,20160526011254Uh,2\n010103-20160703133714Th,20160703133714Th,20160703133714Th,0\n010103-20161224024154Uh,20161224024154Uh,20161222133714Th,1\n010103-20170103030554Uh,20170103030554Uh,20170102024454Uh,0\n010103-20170418013450Uh,20170418013450Uh,20170417014150Uh,1\n010103-20171011170310Mh,20171011170310Mh,20171011170310Mh,1\n010103-20171101133730Ch,20171101133730Ch,20171101133730Ch,2\n010103-20180309000910Mh,20180309000910Mh,20180309000910Mh,0\n010103-20180318020350Uh,20180318020350Uh,20180318020350Uh,2\n010103-20180825205030Ch,20180825205030Ch,20180825205030Ch,0\n010103-20190626025430Lh,20190626025430Lh,20190626025430Lh,2\n010103-20200110073230Lh,20200110073230Lh,20200110073230Lh,1\n010103-20210222222850Bh,20210222222850Bh,20210222222850Bh,1\n010201-20110724164614Mh,20110724164614Mh,20110724164614Mh,0\n010201-20110823120334Ch,20110823120334Ch,20110823120334Ch,1\n010201-20120729122634Ch,20120729122634Ch,20120729122634Ch,2\n010201-20120910074014Th,20120910074014Th,20120910074014Th,1\n010201-20120917142654Bh,20120917142654Bh,20120917142654Bh,1\n010201-20121006063534Lh,20121006063534Lh,20121006063534Lh,0\n010201-20130717123434Ch,20130717123434Ch,20130716162214Mh,1\n010201-20131224192154Bh,20131224192154Bh,20131224192154Bh,0\n010201-20140225163314Th,20140225163314Th,20140225163314Th,0\n010201-20140414020354Uh,20140414020354Uh,20140413195914Mh,2\n010201-20141012015654Uh,20141012015654Uh,20141012015654Uh,1\n010201-20141016022554Uh,20141016022554Uh,20141016022554Uh,1\n010201-20150819082014Th,20150819082014Th,20150819082014Th,0\n010201-20170502233130Lh,20170502233130Lh,20170502233130Lh,1\n010201-20170726235030Lh,20170726235030Lh,20170726235030Lh,0\n010201-20171103222630Lh,20171103222630Lh,20171103222630Lh,2\n010201-20180804234530Lh,20180804234530Lh,20180804234530Lh,1\n010201-20190611032810Mh,20190611032810Mh,20190611032810Mh,1\n010201-20210201020330Lh,20210201020330Lh,20210201020330Lh,2\n010201-20211225025850Uh,20211225025850Uh,20211225025850Uh,1\n010202-20110823120334Ch,20110823120334Ch,20110823120334Ch,1\n010202-20120729122634Ch,20120729122634Ch,20120729122634Ch,2\n010202-20120910074014Th,20120910074014Th,20120910074014Th,1\n010202-20120917142654Bh,20120917142654Bh,20120917142654Bh,1\n010202-20121006063534Lh,20121006063534Lh,20121006063534Lh,0\n010202-20121024164714Mh,20121024164714Mh,20121024164714Mh,0\n010202-20130807073114Th,20130807073114Th,20130807073114Th,2\n010202-20130903141654Bh,20130903141654Bh,20130903060834Lh,0\n010202-20140111224134Lh,20140111224134Lh,20140111224134Lh,1\n010202-20140225163314Th,20140225163314Th,20140225163314Th,0\n010202-20140414020354Uh,20140414020354Uh,20140413195914Mh,2\n010202-20140428233234Lh,20140428233234Lh,20140428233234Lh,0\n010202-20140531195834Ch,20140531195834Ch,20140531195834Ch,1\n010202-20140615195854Bh,20140615195854Bh,20140615195814Mh,0\n010202-20140918195854Bh,20140918195854Bh,20140918195854Bh,2\n010202-20141004075414Th,20141004075414Th,20141004075414Th,0\n010202-20141012015654Uh,20141012015654Uh,20141012015654Uh,1\n010202-20141016022554Uh,20141016022554Uh,20141016022554Uh,1\n010202-20150716162114Mh,20150716162114Mh,20150716162114Mh,0\n010202-20150819082014Th,20150819082014Th,20150819082014Th,0\n010202-20160223173514Mh,20160223173514Mh,20160223173514Mh,2\n010202-20161110151454Bh,20161110151454Bh,20161108021454Uh,0\n010202-20161231133734Ch,20161231133734Ch,20161231133734Ch,0\n010202-20170222133710Th,20170222133710Th,20170222133710Th,2\n010202-20170327143550Bh,20170327143550Bh,20170327143550Bh,0\n010202-20170329133730Ch,20170329133730Ch,20170329133730Ch,2\n010202-20170502233130Lh,20170502233130Lh,20170502233130Lh,1\n010202-20170719235330Lh,20170719235330Lh,20170719235330Lh,1\n010202-20170726235030Lh,20170726235030Lh,20170726235030Lh,0\n010202-20171004143950Bh,20171004143950Bh,20171004143950Bh,2\n010202-20171103222630Lh,20171103222630Lh,20171103222630Lh,2\n010202-20180112130330Ch,20180112130330Ch,20180112130330Ch,1\n010202-20180804234530Lh,20180804234530Lh,20180804234530Lh,1\n010202-20190611032810Mh,20190611032810Mh,20190611032810Mh,1\n010202-20210201020330Lh,20210201020330Lh,20210201020330Lh,2\n010202-20211225025850Uh,20211225025850Uh,20211225025850Uh,1\n010202-20220617185352Mh,20220617185352Mh,20220617185352Mh,2\n010203-20110823120334Ch,20110823120334Ch,20110823120334Ch,1\n010203-20120729122634Ch,20120729122634Ch,20120729122634Ch,2\n010203-20120910074014Th,20120910074014Th,20120910074014Th,1\n010203-20120917142654Bh,20120917142654Bh,20120917142654Bh,1\n010203-20130208063154Uh,20130208063154Uh,20130208063154Uh,1\n010203-20130807073114Th,20130807073114Th,20130807073114Th,2\n010203-20130810121534Ch,20130810121534Ch,20130807073114Th,2\n010203-20130903141654Bh,20130903141654Bh,20130903060834Lh,0\n010203-20130919060834Lh,20130919060834Lh,20130919060834Lh,1\n010203-20140409081614Th,20140409081614Th,20140409081614Th,2\n010203-20140531195834Ch,20140531195834Ch,20140531195834Ch,1\n010203-20141004075414Th,20141004075414Th,20141004075414Th,0\n010203-20141012015654Uh,20141012015654Uh,20141012015654Uh,1\n010203-20141016022554Uh,20141016022554Uh,20141016022554Uh,1\n010203-20141022223334Lh,20141022223334Lh,20141022223334Lh,0\n010203-20150716162114Mh,20150716162114Mh,20150716162114Mh,0\n010203-20150819082014Th,20150819082014Th,20150819082014Th,0\n010203-20160223173514Mh,20160223173514Mh,20160223173514Mh,2\n010203-20161105222634Lh,20161105222634Lh,20161105222634Lh,0\n010203-20161110151454Bh,20161110151454Bh,20161108021454Uh,0\n010203-20161231133734Ch,20161231133734Ch,20161231133734Ch,0\n010203-20170106133714Th,20170106133714Th,20170106133714Th,2\n010203-20170222133710Th,20170222133710Th,20170222133710Th,2\n010203-20170310133730Ch,20170310133730Ch,20170310133730Ch,1\n010203-20170327143550Bh,20170327143550Bh,20170327143550Bh,0\n010203-20170329133730Ch,20170329133730Ch,20170329133730Ch,2\n010203-20170502233130Lh,20170502233130Lh,20170502233130Lh,1\n010203-20170726235030Lh,20170726235030Lh,20170726235030Lh,0\n010203-20171004143950Bh,20171004143950Bh,20171004143950Bh,2\n010203-20180112130330Ch,20180112130330Ch,20180112130330Ch,1\n010203-20180804234530Lh,20180804234530Lh,20180804234530Lh,1\n010203-20190611032810Mh,20190611032810Mh,20190611032810Mh,1\n010203-20210201020330Lh,20210201020330Lh,20210201020330Lh,2\n010203-20210621133650Bh,20210621133650Bh,20210620163530Mh,2\n010203-20211225025850Uh,20211225025850Uh,20211225025850Uh,1\n010203-20211229120210Th,20211229120210Th,20211229120130Ch,0\n010203-20220710085152Th,20220710085152Th,20220710085152Th,1\n010301-20110213093414Th,20110213093414Th,20110213093414Th,2\n010301-20120803102114Th,20120803102114Th,20120802090314Th,0\n010301-20121027063134Lh,20121027063134Lh,20121026063134Lh,2\n010301-20140605195854Bh,20140605195854Bh,20140603070414Th,0\n010301-20151119133734Ch,20151119133734Ch,20151116201114Mh,1\n010301-20160315133714Th,20160315133714Th,20160315133714Th,1\n010301-20160401133734Ch,20160401133734Ch,20160401133734Ch,0\n010301-20180805232510Mh,20180805232510Mh,20180804234530Lh,1\n010301-20181024232430Lh,20181024232430Lh,20181023232430Lh,1\n010301-20190609123330Ch,20190609123330Ch,20190609123330Ch,0\n010302-20110213093414Th,20110213093414Th,20110213093414Th,2\n010302-20111223085314Th,20111223085314Th,20111222063154Uh,2\n010302-20120803102114Th,20120803102114Th,20120802090314Th,0\n010302-20121027063134Lh,20121027063134Lh,20121026063134Lh,2\n010302-20130809072814Th,20130809072814Th,20130807073114Th,2\n010302-20131125192154Bh,20131125192154Bh,20131125192154Bh,1\n010302-20140605195854Bh,20140605195854Bh,20140603070414Th,0\n010302-20151119133734Ch,20151119133734Ch,20151116201114Mh,1\n010302-20151213023454Uh,20151213023454Uh,20151212171154Bh,1\n010302-20160204133734Ch,20160204133734Ch,20160204133734Ch,2\n010302-20160315133714Th,20160315133714Th,20160315133714Th,1\n010302-20170831170750Bh,20170831170750Bh,20170831170750Bh,1\n010302-20180805232510Mh,20180805232510Mh,20180804234530Lh,1\n010302-20180926014350Uh,20180926014350Uh,20180923112330Ch,0\n010302-20190609123330Ch,20190609123330Ch,20190609123330Ch,0\n010302-20210624133950Bh,20210624133950Bh,20210624133950Bh,0\n010303-20160204133734Ch,20160204133734Ch,20160204133734Ch,2\n010303-20170415145450Bh,20170415145450Bh,20170415145450Bh,1\n010303-20170831170750Bh,20170831170750Bh,20170831170750Bh,1\n010303-20180926014350Uh,20180926014350Uh,20180923112330Ch,0\n010303-20210624133950Bh,20210624133950Bh,20210624133950Bh,0\n010401-20110128174814Mh,20110128174814Mh,20110128174814Mh,0\n010401-20110215083814Th,20110215083814Th,20110215083814Th,1\n010401-20110301082654Uh,20110301082654Uh,20110301082654Uh,0\n010401-20110306082634Lh,20110306082634Lh,20110306082634Lh,0\n010401-20110410141654Bh,20110410141654Bh,20110410141654Bh,2\n010401-20110803164914Mh,20110803164914Mh,20110803164914Mh,2\n010401-20110923165814Mh,20110923165814Mh,20110923112234Ch,2\n010401-20111009110234Ch,20111009110234Ch,20111009110234Ch,1\n010401-20111011112054Uh,20111011112054Uh,20111011112054Uh,0\n010401-20111215063134Lh,20111215063134Lh,20111214063134Lh,1\n010401-20111222063154Uh,20111222063154Uh,20111222063154Uh,2\n010401-20111229063154Uh,20111229063154Uh,20111229063154Uh,1\n010401-20120104063134Lh,20120104063134Lh,20120103104134Ch,1\n010401-20120131063154Uh,20120131063154Uh,20120131063154Uh,0\n010401-20120205172414Mh,20120205172414Mh,20120205172414Mh,2\n010401-20120404063154Uh,20120404063154Uh,20120403083114Th,2\n010401-20120424072414Th,20120424072414Th,20120424072414Th,1\n010401-20120519070814Th,20120519070814Th,20120519070814Th,1\n010401-20120610175114Mh,20120610175114Mh,20120610175114Mh,1\n010401-20120703070814Th,20120703070814Th,20120703070814Th,2\n010401-20120723095014Th,20120723095014Th,20120723095014Th,2\n010401-20120802090314Th,20120802090314Th,20120802090314Th,0\n010401-20120913165314Mh,20120913165314Mh,20120913165314Mh,2\n010401-20121113103634Ch,20121113103634Ch,20121113103634Ch,0\n010401-20121129063154Uh,20121129063154Uh,20121129063154Uh,1\n010401-20121218063534Lh,20121218063534Lh,20121218063534Lh,2\n010401-20130221063134Lh,20130221063134Lh,20130221063134Lh,2\n010401-20130303063154Uh,20130303063154Uh,20130303063134Lh,0\n010401-20130319144654Bh,20130319144654Bh,20130319144654Bh,1\n010401-20130517063134Lh,20130517063134Lh,20130517063134Lh,1\n010401-20130813060834Lh,20130813060834Lh,20130813060834Lh,2\n010401-20130925143254Bh,20130925143254Bh,20130925143254Bh,1\n010401-20131007075314Th,20131007075314Th,20131006075314Th,0\n010401-20131011195414Mh,20131011195414Mh,20131011195414Mh,0\n010401-20131027025654Uh,20131027025654Uh,20131027025654Uh,2\n010401-20131107043054Uh,20131107043054Uh,20131107043054Uh,1\n010401-20140205180154Bh,20140205180154Bh,20140205180154Bh,2\n010401-20140216230134Lh,20140216230134Lh,20140216230134Lh,0\n010401-20140406195854Bh,20140406195854Bh,20140406195854Bh,0\n010401-20140413195914Mh,20140413195914Mh,20140413195914Mh,2\n010401-20140509071314Th,20140509071314Th,20140508195834Ch,1\n010401-20141006224634Lh,20141006224634Lh,20141006224634Lh,0\n010401-20141016195834Ch,20141016195834Ch,20141016022554Uh,1\n010401-20141106195854Bh,20141106195854Bh,20141105195814Mh,2\n010401-20141112195854Bh,20141112195854Bh,20141112195854Bh,0\n010401-20150125172714Mh,20150125172714Mh,20150125172714Mh,1\n010401-20150221155154Bh,20150221155154Bh,20150221155154Bh,2\n010401-20150228082414Th,20150228082414Th,20150228082414Th,0\n010401-20150308081954Uh,20150308081954Uh,20150308081954Uh,2\n010401-20150325015554Uh,20150325015554Uh,20150323172814Mh,2\n010401-20150423120034Ch,20150423120034Ch,20150423120034Ch,0\n010401-20150427162314Mh,20150427162314Mh,20150427162314Mh,1\n010401-20150614162514Mh,20150614162514Mh,20150614162514Mh,1\n010401-20150625082034Lh,20150625082034Lh,20150625082014Th,0\n010401-20150825082034Lh,20150825082034Lh,20150825082034Lh,1\n010401-20160213153054Bh,20160213153054Bh,20160213153054Bh,1\n010401-20160215133734Ch,20160215133734Ch,20160215133734Ch,0\n010401-20160225173314Mh,20160225173314Mh,20160225173314Mh,0\n010401-20160229151254Bh,20160229151254Bh,20160229151254Bh,1\n010401-20160403170114Mh,20160403170114Mh,20160401133734Ch,0\n010401-20160810234034Lh,20160810234034Lh,20160808133714Th,0\n010401-20160920230134Lh,20160920230134Lh,20160920230134Lh,2\n010401-20161205133714Th,20161205133714Th,20161205133714Th,0\n010401-20170119031254Uh,20170119031254Uh,20170119031254Uh,2\n010401-20170804140650Bh,20170804140650Bh,20170804140650Bh,1\n010401-20170927143450Bh,20170927143450Bh,20170927143450Bh,0\n010401-20171029150150Bh,20171029150150Bh,20171029150150Bh,0\n010401-20180306081810Th,20180306081810Th,20180306081810Th,2\n010401-20190412025430Lh,20190412025430Lh,20190412025430Lh,2\n010401-20201227202710Mh,20201227202710Mh,20201227202710Mh,1\n010401-20210223222830Ch,20210223222830Ch,20210222222850Bh,1\n010401-20210817165030Mh,20210817165030Mh,20210817165030Mh,2\n010401-20220721185332Bh,20220721185332Bh,20220721185332Bh,1\n010402-20110128174814Mh,20110128174814Mh,20110128174814Mh,0\n010402-20110215083814Th,20110215083814Th,20110215083814Th,1\n010402-20110301082654Uh,20110301082654Uh,20110301082654Uh,0\n010402-20110306082634Lh,20110306082634Lh,20110306082634Lh,0\n010402-20110410141654Bh,20110410141654Bh,20110410141654Bh,2\n010402-20110803164914Mh,20110803164914Mh,20110803164914Mh,2\n010402-20110923165814Mh,20110923165814Mh,20110923112234Ch,2\n010402-20111009110234Ch,20111009110234Ch,20111009110234Ch,1\n010402-20111215063134Lh,20111215063134Lh,20111214063134Lh,1\n010402-20111222063154Uh,20111222063154Uh,20111222063154Uh,2\n010402-20111229063154Uh,20111229063154Uh,20111229063154Uh,1\n010402-20120131063154Uh,20120131063154Uh,20120131063154Uh,0\n010402-20120205172414Mh,20120205172414Mh,20120205172414Mh,2\n010402-20120404063154Uh,20120404063154Uh,20120403083114Th,2\n010402-20120424072414Th,20120424072414Th,20120424072414Th,1\n010402-20120519070814Th,20120519070814Th,20120519070814Th,1\n010402-20120610175114Mh,20120610175114Mh,20120610175114Mh,1\n010402-20120703070814Th,20120703070814Th,20120703070814Th,2\n010402-20120723095014Th,20120723095014Th,20120723095014Th,2\n010402-20120802090314Th,20120802090314Th,20120802090314Th,0\n010402-20120913165314Mh,20120913165314Mh,20120913165314Mh,2\n010402-20121009165914Mh,20121009165914Mh,20121009165914Mh,0\n010402-20121113103634Ch,20121113103634Ch,20121113103634Ch,0\n010402-20121129063154Uh,20121129063154Uh,20121129063154Uh,1\n010402-20121218063534Lh,20121218063534Lh,20121218063534Lh,2\n010402-20130221063134Lh,20130221063134Lh,20130221063134Lh,2\n010402-20130303063154Uh,20130303063154Uh,20130303063134Lh,0\n010402-20130319144654Bh,20130319144654Bh,20130319144654Bh,1\n010402-20130429063154Uh,20130429063154Uh,20130429063154Uh,2\n010402-20130517063134Lh,20130517063134Lh,20130517063134Lh,1\n010402-20130813060834Lh,20130813060834Lh,20130813060834Lh,2\n010402-20130925143254Bh,20130925143254Bh,20130925143254Bh,1\n010402-20131007075314Th,20131007075314Th,20131006075314Th,0\n010402-20131011195414Mh,20131011195414Mh,20131011195414Mh,0\n010402-20131027025654Uh,20131027025654Uh,20131027025654Uh,2\n010402-20131107043054Uh,20131107043054Uh,20131107043054Uh,1\n010402-20140205180154Bh,20140205180154Bh,20140205180154Bh,2\n010402-20140216230134Lh,20140216230134Lh,20140216230134Lh,0\n010402-20140406195854Bh,20140406195854Bh,20140406195854Bh,0\n010402-20140509071314Th,20140509071314Th,20140508195834Ch,1\n010402-20141006224634Lh,20141006224634Lh,20141006224634Lh,0\n010402-20141016195834Ch,20141016195834Ch,20141016022554Uh,1\n010402-20141106195854Bh,20141106195854Bh,20141105195814Mh,2\n010402-20141112195854Bh,20141112195854Bh,20141112195854Bh,0\n010402-20150125172714Mh,20150125172714Mh,20150125172714Mh,1\n010402-20150221155154Bh,20150221155154Bh,20150221155154Bh,2\n010402-20150228082414Th,20150228082414Th,20150228082414Th,0\n010402-20150308081954Uh,20150308081954Uh,20150308081954Uh,2\n010402-20150325015554Uh,20150325015554Uh,20150323172814Mh,2\n010402-20150423120034Ch,20150423120034Ch,20150423120034Ch,0\n010402-20150427162314Mh,20150427162314Mh,20150427162314Mh,1\n010402-20150614162514Mh,20150614162514Mh,20150614162514Mh,1\n010402-20150625082034Lh,20150625082034Lh,20150625082014Th,0\n010402-20150825082034Lh,20150825082034Lh,20150825082034Lh,1\n010402-20160213153054Bh,20160213153054Bh,20160213153054Bh,1\n010402-20160215133734Ch,20160215133734Ch,20160215133734Ch,0\n010402-20160225173314Mh,20160225173314Mh,20160225173314Mh,0\n010402-20160229151254Bh,20160229151254Bh,20160229151254Bh,1\n010402-20160403170114Mh,20160403170114Mh,20160401133734Ch,0\n010402-20160430133714Th,20160430133714Th,20160430133714Th,0\n010402-20160810234034Lh,20160810234034Lh,20160808133714Th,0\n010402-20160920230134Lh,20160920230134Lh,20160920230134Lh,2\n010402-20161205133714Th,20161205133714Th,20161205133714Th,0\n010402-20170119031254Uh,20170119031254Uh,20170119031254Uh,2\n010402-20170804140650Bh,20170804140650Bh,20170804140650Bh,1\n010402-20170927143450Bh,20170927143450Bh,20170927143450Bh,0\n010402-20171007133730Ch,20171007133730Ch,20171007133710Th,1\n010402-20171029150150Bh,20171029150150Bh,20171029150150Bh,0\n010402-20180306081810Th,20180306081810Th,20180306081810Th,2\n010402-20190412025430Lh,20190412025430Lh,20190412025430Lh,2\n010402-20201227202710Mh,20201227202710Mh,20201227202710Mh,1\n010402-20210223222830Ch,20210223222830Ch,20210222222850Bh,1\n010402-20210817165030Mh,20210817165030Mh,20210817165030Mh,2\n010402-20220418020950Uh,20220418020950Uh,20220418020950Uh,2\n010402-20220721185332Bh,20220721185332Bh,20220721185332Bh,1\n010403-20110128174814Mh,20110128174814Mh,20110128174814Mh,0\n010403-20110215083814Th,20110215083814Th,20110215083814Th,1\n010403-20110301082654Uh,20110301082654Uh,20110301082654Uh,0\n010403-20110306082634Lh,20110306082634Lh,20110306082634Lh,0\n010403-20110410141654Bh,20110410141654Bh,20110410141654Bh,2\n010403-20110803164914Mh,20110803164914Mh,20110803164914Mh,2\n010403-20110923165814Mh,20110923165814Mh,20110923112234Ch,2\n010403-20111009110234Ch,20111009110234Ch,20111009110234Ch,1\n010403-20111215063134Lh,20111215063134Lh,20111214063134Lh,1\n010403-20111222063154Uh,20111222063154Uh,20111222063154Uh,2\n010403-20111229063154Uh,20111229063154Uh,20111229063154Uh,1\n010403-20120104063134Lh,20120104063134Lh,20120103104134Ch,1\n010403-20120131063154Uh,20120131063154Uh,20120131063154Uh,0\n010403-20120205172414Mh,20120205172414Mh,20120205172414Mh,2\n010403-20120404063154Uh,20120404063154Uh,20120403083114Th,2\n010403-20120424072414Th,20120424072414Th,20120424072414Th,1\n010403-20120519070814Th,20120519070814Th,20120519070814Th,1\n010403-20120610175114Mh,20120610175114Mh,20120610175114Mh,1\n010403-20120703070814Th,20120703070814Th,20120703070814Th,2\n010403-20120723095014Th,20120723095014Th,20120723095014Th,2\n010403-20120802090314Th,20120802090314Th,20120802090314Th,0\n010403-20120913165314Mh,20120913165314Mh,20120913165314Mh,2\n010403-20121009165914Mh,20121009165914Mh,20121009165914Mh,0\n010403-20121113103634Ch,20121113103634Ch,20121113103634Ch,0\n010403-20121129063154Uh,20121129063154Uh,20121129063154Uh,1\n010403-20121218063534Lh,20121218063534Lh,20121218063534Lh,2\n010403-20130221063134Lh,20130221063134Lh,20130221063134Lh,2\n010403-20130303063154Uh,20130303063154Uh,20130303063134Lh,0\n010403-20130319144654Bh,20130319144654Bh,20130319144654Bh,1\n010403-20130429063154Uh,20130429063154Uh,20130429063154Uh,2\n010403-20130517063134Lh,20130517063134Lh,20130517063134Lh,1\n010403-20130813060834Lh,20130813060834Lh,20130813060834Lh,2\n010403-20130925143254Bh,20130925143254Bh,20130925143254Bh,1\n010403-20131007075314Th,20131007075314Th,20131006075314Th,0\n010403-20131011195414Mh,20131011195414Mh,20131011195414Mh,0\n010403-20131027025654Uh,20131027025654Uh,20131027025654Uh,2\n010403-20131107043054Uh,20131107043054Uh,20131107043054Uh,1\n010403-20140205180154Bh,20140205180154Bh,20140205180154Bh,2\n010403-20140216230134Lh,20140216230134Lh,20140216230134Lh,0\n010403-20140406195854Bh,20140406195854Bh,20140406195854Bh,0\n010403-20140413195914Mh,20140413195914Mh,20140413195914Mh,2\n010403-20140509071314Th,20140509071314Th,20140508195834Ch,1\n010403-20141006224634Lh,20141006224634Lh,20141006224634Lh,0\n010403-20141016195834Ch,20141016195834Ch,20141016022554Uh,1\n010403-20141106195854Bh,20141106195854Bh,20141105195814Mh,2\n010403-20141112195854Bh,20141112195854Bh,20141112195854Bh,0\n010403-20150125172714Mh,20150125172714Mh,20150125172714Mh,1\n010403-20150221155154Bh,20150221155154Bh,20150221155154Bh,2\n010403-20150228082414Th,20150228082414Th,20150228082414Th,0\n010403-20150308081954Uh,20150308081954Uh,20150308081954Uh,2\n010403-20150325015554Uh,20150325015554Uh,20150323172814Mh,2\n010403-20150423120034Ch,20150423120034Ch,20150423120034Ch,0\n010403-20150427162314Mh,20150427162314Mh,20150427162314Mh,1\n010403-20150614162514Mh,20150614162514Mh,20150614162514Mh,1\n010403-20150625082034Lh,20150625082034Lh,20150625082014Th,0\n010403-20150825082034Lh,20150825082034Lh,20150825082034Lh,1\n010403-20160213153054Bh,20160213153054Bh,20160213153054Bh,1\n010403-20160215133734Ch,20160215133734Ch,20160215133734Ch,0\n010403-20160225173314Mh,20160225173314Mh,20160225173314Mh,0\n010403-20160229151254Bh,20160229151254Bh,20160229151254Bh,1\n010403-20160403170114Mh,20160403170114Mh,20160401133734Ch,0\n010403-20160430133714Th,20160430133714Th,20160430133714Th,0\n010403-20160810234034Lh,20160810234034Lh,20160808133714Th,0\n010403-20160920230134Lh,20160920230134Lh,20160920230134Lh,2\n010403-20161205133714Th,20161205133714Th,20161205133714Th,0\n010403-20170119031254Uh,20170119031254Uh,20170119031254Uh,2\n010403-20170804140650Bh,20170804140650Bh,20170804140650Bh,1\n010403-20170927143450Bh,20170927143450Bh,20170927143450Bh,0\n010403-20171007133730Ch,20171007133730Ch,20171007133710Th,1\n010403-20171029150150Bh,20171029150150Bh,20171029150150Bh,0\n010403-20180306081810Th,20180306081810Th,20180306081810Th,2\n010403-20190412025430Lh,20190412025430Lh,20190412025430Lh,2\n010403-20201227202710Mh,20201227202710Mh,20201227202710Mh,1\n010403-20210223222830Ch,20210223222830Ch,20210222222850Bh,1\n010403-20210626060050Uh,20210626060050Uh,20210626060050Uh,1\n010403-20210817165030Mh,20210817165030Mh,20210817165030Mh,2\n010403-20220721185332Bh,20220721185332Bh,20220721185332Bh,1\n020101-20110406082634Lh,20110406082634Lh,20110406082634Lh,2\n020101-20110521063134Lh,20110521063134Lh,20110520063154Uh,0\n020101-20110526122534Ch,20110526122534Ch,20110525063154Uh,2\n020101-20110619063134Lh,20110619063134Lh,20110619063134Lh,0\n020101-20120330063154Uh,20120330063154Uh,20120330063134Lh,0\n020101-20120827085114Th,20120827085114Th,20120827085114Th,2\n020101-20121022145554Bh,20121022145554Bh,20121021145454Bh,1\n020101-20130303063134Lh,20130303063134Lh,20130303063134Lh,0\n020101-20130405114734Ch,20130405114734Ch,20130404063134Lh,2\n020101-20130609123334Ch,20130609123334Ch,20130609123334Ch,1\n020101-20130905073714Th,20130905073714Th,20130905073714Th,1\n020101-20131020080114Th,20131020080114Th,20131019080114Th,0\n020101-20131116185334Ch,20131116185334Ch,20131115185414Mh,0\n020101-20131204192154Bh,20131204192154Bh,20131204192154Bh,2\n020101-20140430071914Th,20140430071914Th,20140430071914Th,1\n020101-20140525070514Th,20140525070514Th,20140525070514Th,0\n020101-20140908175354Bh,20140908175354Bh,20140908175354Bh,2\n020101-20150330081954Uh,20150330081954Uh,20150330081954Uh,1\n020101-20150630083214Th,20150630083214Th,20150629161614Mh,2\n020101-20150820152254Bh,20150820152254Bh,20150819082014Th,0\n020101-20151106171514Mh,20151106171514Mh,20151105133734Ch,0\n020101-20151212171154Bh,20151212171154Bh,20151212171154Bh,1\n020101-20160201025254Uh,20160201025254Uh,20160201025254Uh,2\n020101-20160309172214Mh,20160309172214Mh,20160309172214Mh,2\n020101-20160518011654Uh,20160518011654Uh,20160518011654Uh,1\n020101-20161013133734Ch,20161013133734Ch,20161012133734Ch,2\n020101-20161109171714Mh,20161109171714Mh,20161108021454Uh,0\n020101-20170411013550Uh,20170411013550Uh,20170411013550Uh,2\n020101-20170701235630Lh,20170701235630Lh,20170701235630Lh,2\n020101-20170722235230Lh,20170722235230Lh,20170722235230Lh,2\n020101-20170801164710Mh,20170801164710Mh,20170801164710Mh,2\n020101-20180925000030Lh,20180925000030Lh,20180923112330Ch,0\n020101-20190927165910Mh,20190927165910Mh,20190925165810Mh,2\n020101-20200105073250Uh,20200105073250Uh,20200105073250Uh,0\n020101-20200619123010Th,20200619123010Th,20200619123010Th,0\n020101-20201022223330Lh,20201022223330Lh,20201022223330Lh,2\n020101-20210812044930Lh,20210812044930Lh,20210812044930Lh,0\n020101-20220718185352Mh,20220718185352Mh,20220718185352Mh,1\n020103-20150330081954Uh,20150330081954Uh,20150330081954Uh,1\n020201-20110520063154Uh,20110520063154Uh,20110520063154Uh,0\n020401-20110205082634Lh,20110205082634Lh,20110205082634Lh,2\n020401-20110324082614Th,20110324082614Th,20110324082614Th,2\n020401-20110510063134Lh,20110510063134Lh,20110509121334Ch,0\n020401-20110616133454Bh,20110616133454Bh,20110615133454Bh,1\n020401-20110707074514Th,20110707074514Th,20110707074514Th,2\n020401-20110815140454Bh,20110815140454Bh,20110815140454Bh,0\n020401-20111018063134Lh,20111018063134Lh,20111018063134Lh,2\n020401-20111112193254Bh,20111112193254Bh,20111112193254Bh,1\n020401-20120217181154Bh,20120217181154Bh,20120217063154Uh,1\n020401-20120223063154Uh,20120223063154Uh,20120223063154Uh,1\n020401-20120225172914Mh,20120225172914Mh,20120225172914Mh,0\n020401-20120318063154Uh,20120318063154Uh,20120318063154Uh,1\n020401-20120331142954Bh,20120331142954Bh,20120330063134Lh,0\n020401-20120403164114Mh,20120403164114Mh,20120403083114Th,2\n020401-20120814163114Mh,20120814163114Mh,20120813164714Mh,1\n020401-20121114172254Bh,20121114172254Bh,20121113103634Ch,0\n020401-20130110104734Ch,20130110104734Ch,20130109130334Ch,2\n020401-20130219063134Lh,20130219063134Lh,20130218220314Mh,0\n020401-20130314165714Mh,20130314165714Mh,20130314165714Mh,2\n020401-20130923060834Lh,20130923060834Lh,20130923060834Lh,2\n020401-20131101222734Lh,20131101222734Lh,20131030185354Bh,0\n020401-20131206084114Th,20131206084114Th,20131206084114Th,0\n020401-20140212024554Uh,20140212024554Uh,20140212024554Uh,1\n020401-20140806195834Ch,20140806195834Ch,20140806195834Ch,1\n020401-20141013195834Ch,20141013195834Ch,20141012015654Uh,1\n020401-20141105195814Mh,20141105195814Mh,20141105195814Mh,2\n020401-20141225024354Uh,20141225024354Uh,20141225024354Uh,1\n020401-20150610082014Th,20150610082014Th,20150610082014Th,2\n020401-20150617161314Mh,20150617161314Mh,20150617161314Mh,2\n020401-20150706083014Th,20150706083014Th,20150706083014Th,0\n020401-20150720082014Th,20150720082014Th,20150718082014Th,1\n020401-20160402025254Uh,20160402025254Uh,20160401133734Ch,0\n020401-20160501143154Bh,20160501143154Bh,20160430133714Th,0\n020401-20160628152654Bh,20160628152654Bh,20160627133734Ch,2\n020401-20170417014150Uh,20170417014150Uh,20170417014150Uh,1\n020401-20170425133730Ch,20170425133730Ch,20170425133730Ch,2\n020401-20171007133710Th,20171007133710Th,20171007133710Th,1\n020401-20171022170710Mh,20171022170710Mh,20171022170710Mh,1\n020401-20171025223130Lh,20171025223130Lh,20171025133730Ch,0\n020401-20171106171510Mh,20171106171510Mh,20171106171510Mh,2\n020401-20180318232230Lh,20180318232230Lh,20180318020350Uh,2\n020401-20200114073230Lh,20200114073230Lh,20200114073230Lh,0\n020401-20220122085210Th,20220122085210Th,20220122085210Th,1\n020403-20110205082634Lh,20110205082634Lh,20110205082634Lh,2\n020403-20110324082614Th,20110324082614Th,20110324082614Th,2\n020403-20110510063134Lh,20110510063134Lh,20110509121334Ch,0\n020403-20110616133454Bh,20110616133454Bh,20110615133454Bh,1\n020403-20110707074514Th,20110707074514Th,20110707074514Th,2\n020403-20110815140454Bh,20110815140454Bh,20110815140454Bh,0\n020403-20111018063134Lh,20111018063134Lh,20111018063134Lh,2\n020403-20111112193254Bh,20111112193254Bh,20111112193254Bh,1\n020403-20120217181154Bh,20120217181154Bh,20120217063154Uh,1\n020403-20120223063154Uh,20120223063154Uh,20120223063154Uh,1\n020403-20120225172914Mh,20120225172914Mh,20120225172914Mh,0\n020403-20120318063154Uh,20120318063154Uh,20120318063154Uh,1\n020403-20120331142954Bh,20120331142954Bh,20120330063134Lh,0\n020403-20120403164114Mh,20120403164114Mh,20120403083114Th,2\n020403-20120814163114Mh,20120814163114Mh,20120813164714Mh,1\n020403-20121114172254Bh,20121114172254Bh,20121113103634Ch,0\n020403-20130110104734Ch,20130110104734Ch,20130109130334Ch,2\n020403-20130219063134Lh,20130219063134Lh,20130218220314Mh,0\n020403-20130314165714Mh,20130314165714Mh,20130314165714Mh,2\n020403-20130923060834Lh,20130923060834Lh,20130923060834Lh,2\n020403-20131101222734Lh,20131101222734Lh,20131030185354Bh,0\n020403-20131206084114Th,20131206084114Th,20131206084114Th,0\n020403-20140212024554Uh,20140212024554Uh,20140212024554Uh,1\n020403-20140806195834Ch,20140806195834Ch,20140806195834Ch,1\n020403-20141013195834Ch,20141013195834Ch,20141012015654Uh,1\n020403-20141105195814Mh,20141105195814Mh,20141105195814Mh,2\n020403-20141225024354Uh,20141225024354Uh,20141225024354Uh,1\n020403-20150610082014Th,20150610082014Th,20150610082014Th,2\n020403-20150617161314Mh,20150617161314Mh,20150617161314Mh,2\n020403-20150706083014Th,20150706083014Th,20150706083014Th,0\n020403-20150720082014Th,20150720082014Th,20150718082014Th,1\n020403-20160402025254Uh,20160402025254Uh,20160401133734Ch,0\n020403-20160501143154Bh,20160501143154Bh,20160430133714Th,0\n020403-20160628152654Bh,20160628152654Bh,20160627133734Ch,2\n020403-20170417014150Uh,20170417014150Uh,20170417014150Uh,1\n020403-20170425133730Ch,20170425133730Ch,20170425133730Ch,2\n020403-20171022170710Mh,20171022170710Mh,20171022170710Mh,1\n020403-20171025223130Lh,20171025223130Lh,20171025133730Ch,0\n020403-20171106171510Mh,20171106171510Mh,20171106171510Mh,2\n020403-20180318232230Lh,20180318232230Lh,20180318020350Uh,2\n020403-20200114073230Lh,20200114073230Lh,20200114073230Lh,0\n020403-20220122085210Th,20220122085210Th,20220122085210Th,1\n030101-20110109104734Ch,20110109104734Ch,20110109104734Ch,2\n030101-20110317082654Uh,20110317082654Uh,20110317082654Uh,0\n030101-20110601063134Lh,20110601063134Lh,20110601063134Lh,1\n030101-20110617063134Lh,20110617063134Lh,20110615133454Bh,1\n030101-20110623063134Lh,20110623063134Lh,20110623063134Lh,2\n030101-20110902073614Th,20110902073614Th,20110902073614Th,1\n030101-20110923112234Ch,20110923112234Ch,20110923112234Ch,2\n030101-20120313063154Uh,20120313063154Uh,20120313063154Uh,2\n030101-20120323144054Bh,20120323144054Bh,20120323144054Bh,2\n030101-20120403083114Th,20120403083114Th,20120403083114Th,2\n030101-20121118092514Th,20121118092514Th,20121118092514Th,0\n030101-20130111104734Ch,20130111104734Ch,20130109130334Ch,2\n030101-20130201084914Th,20130201084914Th,20130201084914Th,0\n030101-20130225171214Mh,20130225171214Mh,20130225171214Mh,1\n030101-20130510134654Bh,20130510134654Bh,20130510134654Bh,1\n030101-20130615002434Lh,20130615002434Lh,20130615002434Lh,1\n030101-20130725122934Ch,20130725122934Ch,20130725122934Ch,0\n030101-20130805135854Bh,20130805135854Bh,20130805135854Bh,1\n030101-20130926060834Lh,20130926060834Lh,20130925143254Bh,1\n030101-20131031185614Mh,20131031185614Mh,20131030185354Bh,0\n030101-20140323020654Uh,20140323020654Uh,20140323020654Uh,2\n030101-20140719071414Th,20140719071414Th,20140719071414Th,2\n030101-20140720195834Ch,20140720195834Ch,20140719071414Th,2\n030101-20140816072814Th,20140816072814Th,20140816072814Th,1\n030101-20141013195854Bh,20141013195854Bh,20141012015654Uh,1\n030101-20141116222234Lh,20141116222234Lh,20141116222234Lh,1\n030101-20150130084914Th,20150130084914Th,20150130084914Th,2\n030101-20150203084714Th,20150203084714Th,20150203084714Th,1\n030101-20150501120634Ch,20150501120634Ch,20150501120634Ch,2\n030101-20150615133554Bh,20150615133554Bh,20150614162514Mh,1\n030101-20150719182314Mh,20150719182314Mh,20150718082014Th,1\n030101-20150919165814Mh,20150919165814Mh,20150919165814Mh,0\n030101-20151013144654Bh,20151013144654Bh,20151013144654Bh,1\n030101-20160103133734Ch,20160103133734Ch,20160103133734Ch,2\n030101-20160127225034Lh,20160127225034Lh,20160127225034Lh,1\n030101-20160317232434Lh,20160317232434Lh,20160317232434Lh,0\n030101-20160408232234Lh,20160408232234Lh,20160408232234Lh,1\n030101-20160411134234Ch,20160411134234Ch,20160411134234Ch,2\n030101-20160516012454Uh,20160516012454Uh,20160516012454Uh,1\n030101-20160609133654Bh,20160609133654Bh,20160609133654Bh,1\n030101-20160726135354Bh,20160726135354Bh,20160726135354Bh,1\n030101-20160831133714Th,20160831133714Th,20160831133714Th,2\n030101-20161116133714Th,20161116133714Th,20161116133714Th,0\n030101-20170427154550Bh,20170427154550Bh,20170427154550Bh,2\n030101-20171005133730Ch,20171005133730Ch,20171004143950Bh,2\n030101-20190925165810Mh,20190925165810Mh,20190925165810Mh,2\n030101-20210103223630Lh,20210103223630Lh,20210103223630Lh,1\n030101-20210815140550Bh,20210815140550Bh,20210814044930Lh,0\n030302-20110325171014Mh,20110325171014Mh,20110324082614Th,2\n030302-20110421164914Mh,20110421164914Mh,20110421164914Mh,0\n030302-20110530122834Ch,20110530122834Ch,20110530122834Ch,0\n030302-20110603163414Mh,20110603163414Mh,20110603163414Mh,2\n030302-20111107063154Uh,20111107063154Uh,20111107063154Uh,1\n030302-20120103104134Ch,20120103104134Ch,20120103104134Ch,1\n030302-20120227161654Bh,20120227161654Bh,20120227161654Bh,0\n030302-20120306063154Uh,20120306063154Uh,20120306063154Uh,0\n030302-20120424120134Ch,20120424120134Ch,20120424072414Th,1\n030302-20120712123634Ch,20120712123634Ch,20120712123634Ch,2\n030302-20120928063134Lh,20120928063134Lh,20120928063134Lh,1\n030302-20121026063134Lh,20121026063134Lh,20121026063134Lh,2\n030302-20130324015554Uh,20130324015554Uh,20130324015554Uh,2\n030302-20130609173414Mh,20130609173414Mh,20130609123334Ch,1\n030302-20131030185354Bh,20131030185354Bh,20131030185354Bh,0\n030302-20131109190514Mh,20131109190514Mh,20131107043054Uh,1\n030302-20131113185414Mh,20131113185414Mh,20131113004514Mh,1\n030302-20140501195814Mh,20140501195814Mh,20140430071914Th,1\n030302-20140523195814Mh,20140523195814Mh,20140522070614Th,1\n030302-20140615195814Mh,20140615195814Mh,20140615195814Mh,0\n030302-20140712195854Bh,20140712195854Bh,20140712195854Bh,1\n030302-20140725195814Mh,20140725195814Mh,20140725195814Mh,0\n030302-20140726071814Th,20140726071814Th,20140725195814Mh,0\n030302-20140921195854Bh,20140921195854Bh,20140921195854Bh,0\n030302-20141009023454Uh,20141009023454Uh,20141009023454Uh,2\n030302-20141025031154Uh,20141025031154Uh,20141024195834Ch,2\n030302-20150302082034Lh,20150302082034Lh,20150302082034Lh,2\n030302-20150625082014Th,20150625082014Th,20150625082014Th,0\n030302-20150718082014Th,20150718082014Th,20150718082014Th,1\n030302-20151011144554Bh,20151011144554Bh,20151011144554Bh,0\n030302-20151128222134Lh,20151128222134Lh,20151128133714Th,1\n030302-20160125154854Bh,20160125154854Bh,20160125154854Bh,2\n030302-20160510163914Mh,20160510163914Mh,20160510163914Mh,1\n030302-20160627133734Ch,20160627133734Ch,20160627133734Ch,2\n030302-20161108021454Uh,20161108021454Uh,20161108021454Uh,0\n030302-20161210133714Th,20161210133714Th,20161210133714Th,0\n030302-20170323041730Lh,20170323041730Lh,20170322015550Uh,0\n030302-20170408133710Th,20170408133710Th,20170408133710Th,2\n030302-20170417232530Lh,20170417232530Lh,20170417014150Uh,1\n030302-20170619012150Uh,20170619012150Uh,20170619012150Uh,1\n030302-20171025133730Ch,20171025133730Ch,20171025133730Ch,0\n030302-20171121222130Lh,20171121222130Lh,20171121222130Lh,0\n030302-20190629163710Mh,20190629163710Mh,20190629163710Mh,1\n030401-20111214132434Ch,20111214132434Ch,20111214063134Lh,1\n030401-20130407063154Uh,20130407063154Uh,20130407063154Uh,2\n030401-20140401014754Uh,20140401014754Uh,20140401014754Uh,1\n030401-20140922195854Bh,20140922195854Bh,20140921195854Bh,0\n030401-20150727082014Th,20150727082014Th,20150727082014Th,1\n030401-20151224133734Ch,20151224133734Ch,20151224133734Ch,1\n030401-20170925225630Lh,20170925225630Lh,20170925225630Lh,1\n030401-20211231223430Lh,20211231223430Lh,20211231223430Lh,2\n030403-20111115063134Lh,20111115063134Lh,20111114082314Th,2\n030403-20111214132434Ch,20111214132434Ch,20111214063134Lh,1\n030403-20120907141954Bh,20120907141954Bh,20120907141954Bh,2\n030403-20121108155954Bh,20121108155954Bh,20121108155954Bh,2\n030403-20130407063154Uh,20130407063154Uh,20130407063154Uh,2\n030403-20140401014754Uh,20140401014754Uh,20140401014754Uh,1\n030403-20140723071614Th,20140723071614Th,20140723071614Th,2\n030403-20150727082014Th,20150727082014Th,20150727082014Th,1\n030403-20170423014350Uh,20170423014350Uh,20170423014350Uh,0\n030403-20180506140230Ch,20180506140230Ch,20180506140230Ch,0\n030403-20210810121630Ch,20210810121630Ch,20210810121630Ch,1\n030403-20211231223430Lh,20211231223430Lh,20211231223430Lh,2\n040102-20110511063154Uh,20110511063154Uh,20110509121334Ch,0\n040102-20110525063154Uh,20110525063154Uh,20110525063154Uh,2\n040102-20110824141054Bh,20110824141054Bh,20110823120334Ch,1\n040102-20111214063134Lh,20111214063134Lh,20111214063134Lh,1\n040102-20120116063254Uh,20120116063254Uh,20120116063254Uh,2\n040102-20120427162614Mh,20120427162614Mh,20120427063154Uh,0\n040102-20120726071914Th,20120726071914Th,20120725122934Ch,1\n040102-20121207103034Ch,20121207103034Ch,20121207103034Ch,0\n040102-20130214111534Ch,20130214111534Ch,20130214111534Ch,2\n040102-20130326063134Lh,20130326063134Lh,20130326063134Lh,0\n040102-20130921060834Lh,20130921060834Lh,20130921060834Lh,2\n040102-20131201192134Ch,20131201192134Ch,20131130192134Ch,1\n040102-20140312025454Uh,20140312025454Uh,20140312025454Uh,0\n040102-20140603070414Th,20140603070414Th,20140603070414Th,0\n040102-20140607112354Uh,20140607112354Uh,20140607112354Uh,1\n040102-20140607195854Bh,20140607195854Bh,20140607112354Uh,1\n040102-20140802072114Th,20140802072114Th,20140802072114Th,1\n040102-20141007224934Lh,20141007224934Lh,20141006224634Lh,0\n040102-20141024195834Ch,20141024195834Ch,20141024195834Ch,2\n040102-20150119082034Lh,20150119082034Lh,20150119082034Lh,2\n040102-20150225171114Mh,20150225171114Mh,20150225171114Mh,0\n040102-20150315082034Lh,20150315082034Lh,20150314081954Uh,2\n040102-20151118172314Mh,20151118172314Mh,20151116201114Mh,1\n040102-20160129225334Lh,20160129225334Lh,20160129225334Lh,2\n040102-20160302133734Ch,20160302133734Ch,20160302133734Ch,0\n040102-20160604163414Mh,20160604163414Mh,20160604163414Mh,0\n040102-20161002225034Lh,20161002225034Lh,20161002225034Lh,2\n040102-20161201160954Bh,20161201160954Bh,20161201160954Bh,0\n040102-20161217133734Ch,20161217133734Ch,20161217133734Ch,2\n040102-20161223133734Ch,20161223133734Ch,20161222133714Th,1\n040102-20170702133730Ch,20170702133730Ch,20170701235630Lh,2\n040102-20171011224130Lh,20171011224130Lh,20171011170310Mh,1\n040102-20171109023950Uh,20171109023950Uh,20171109023950Uh,1\n040102-20171123133730Ch,20171123133730Ch,20171123133730Ch,0\n040102-20180710031950Uh,20180710031950Uh,20180710031950Uh,0\n040102-20200806053830Lh,20200806053830Lh,20200806053830Lh,1\n040102-20210731074710Th,20210731074710Th,20210731074710Th,1\n040102-20211229120130Ch,20211229120130Ch,20211229120130Ch,0\n040102-20220803074752Th,20220803074752Th,20220803074752Th,1\n040201-20111019063154Uh,20111019063154Uh,20111018063134Lh,2\n040201-20120725122934Ch,20120725122934Ch,20120725122934Ch,1\n040201-20130116172814Mh,20130116172814Mh,20130116172814Mh,2\n040201-20130306172214Mh,20130306172214Mh,20130303063134Lh,0\n040201-20130404063134Lh,20130404063134Lh,20130404063134Lh,2\n040201-20130611063154Uh,20130611063154Uh,20130611063154Uh,1\n040201-20130705161714Mh,20130705161714Mh,20130705161714Mh,2\n040201-20130716162214Mh,20130716162214Mh,20130716162214Mh,1\n040201-20140420085614Th,20140420085614Th,20140420085614Th,1\n040201-20140601195814Mh,20140601195814Mh,20140531195834Ch,1\n040201-20140821195834Ch,20140821195834Ch,20140821195834Ch,0\n040201-20140903073614Th,20140903073614Th,20140903073614Th,2\n040201-20160215175514Mh,20160215175514Mh,20160215133734Ch,0\n040201-20160808133714Th,20160808133714Th,20160808133714Th,0\n040201-20171014133730Ch,20171014133730Ch,20171014133730Ch,0\n040201-20171114133730Ch,20171114133730Ch,20171114133730Ch,2\n040201-20190704123830Ch,20190704123830Ch,20190704123830Ch,1\n040201-20210620163530Mh,20210620163530Mh,20210620163530Mh,2\n040201-20210728164530Mh,20210728164530Mh,20210728164530Mh,2\n040201-20210814044930Lh,20210814044930Lh,20210814044930Lh,0\n040301-20110711071214Th,20110711071214Th,20110711071214Th,1\n040301-20111201063154Uh,20111201063154Uh,20111201063154Uh,0\n040301-20111206102934Ch,20111206102934Ch,20111206102934Ch,2\n040301-20111208063134Lh,20111208063134Lh,20111208063134Lh,0\n040301-20120217063154Uh,20120217063154Uh,20120217063154Uh,1\n040301-20120218173614Mh,20120218173614Mh,20120217063154Uh,1\n040301-20120223152054Bh,20120223152054Bh,20120223063154Uh,1\n040301-20120328164614Mh,20120328164614Mh,20120328164614Mh,0\n040301-20120414175914Mh,20120414175914Mh,20120414115434Ch,0\n040301-20120430063154Uh,20120430063154Uh,20120430063154Uh,2\n040301-20120901114934Ch,20120901114934Ch,20120901114934Ch,2\n040301-20120914063134Lh,20120914063134Lh,20120913165314Mh,2\n040301-20130109130334Ch,20130109130334Ch,20130109130334Ch,2\n040301-20130211063134Lh,20130211063134Lh,20130211063134Lh,0\n040301-20130304170614Mh,20130304170614Mh,20130303063134Lh,0\n040301-20130502063134Lh,20130502063134Lh,20130502063134Lh,0\n040301-20130828141354Bh,20130828141354Bh,20130828141354Bh,2\n040301-20130913142354Bh,20130913142354Bh,20130913142354Bh,2\n040301-20131113004514Mh,20131113004514Mh,20131113004514Mh,1\n040301-20131130192134Ch,20131130192134Ch,20131130192134Ch,1\n040301-20140213024554Uh,20140213024554Uh,20140212024554Uh,1\n040301-20140218230234Lh,20140218230234Lh,20140218230234Lh,0\n040301-20140508195834Ch,20140508195834Ch,20140508195834Ch,1\n040301-20140609195854Bh,20140609195854Bh,20140609195854Bh,2\n040301-20140701195814Mh,20140701195814Mh,20140701195814Mh,1\n040301-20140807195854Bh,20140807195854Bh,20140806195834Ch,1\n040301-20140817195834Ch,20140817195834Ch,20140816072814Th,1\n040301-20140904195814Mh,20140904195814Mh,20140903073614Th,2\n040301-20150222090014Th,20150222090014Th,20150221155154Bh,2\n040301-20150222112034Ch,20150222112034Ch,20150221155154Bh,2\n040301-20151128133734Ch,20151128133734Ch,20151128133714Th,1\n040301-20151226133734Ch,20151226133734Ch,20151226133734Ch,2\n040301-20161012133734Ch,20161012133734Ch,20161012133734Ch,2\n040301-20170122034654Uh,20170122034654Uh,20170122034654Uh,0\n040301-20170612135150Bh,20170612135150Bh,20170612135150Bh,1\n040301-20180109223930Lh,20180109223930Lh,20180109223930Lh,0\n040301-20190701123830Ch,20190701123830Ch,20190701123830Ch,1\n040301-20210331231510Mh,20210331231510Mh,20210331231510Mh,1\n040301-20210626133950Bh,20210626133950Bh,20210626060050Uh,1\n040302-20110711071214Th,20110711071214Th,20110711071214Th,1\n040302-20120914063134Lh,20120914063134Lh,20120913165314Mh,2\n040302-20140508195834Ch,20140508195834Ch,20140508195834Ch,1\n040302-20140807195854Bh,20140807195854Bh,20140806195834Ch,1\n040302-20151128133734Ch,20151128133734Ch,20151128133714Th,1\n040302-20190701123830Ch,20190701123830Ch,20190701123830Ch,1\n040302-20210329015250Uh,20210329015250Uh,20210329015250Uh,0\n040302-20210331231510Mh,20210331231510Mh,20210331231510Mh,1\n040302-20220104043050Uh,20220104043050Uh,20220104043050Uh,0\n040303-20130828141354Bh,20130828141354Bh,20130828141354Bh,2\n040303-20140218230234Lh,20140218230234Lh,20140218230234Lh,0\n040303-20210626133950Bh,20210626133950Bh,20210626060050Uh,1\n040401-20110119082634Lh,20110119082634Lh,20110119082634Lh,0\n040401-20110312082634Lh,20110312082634Lh,20110312082634Lh,2\n040401-20110509121334Ch,20110509121334Ch,20110509121334Ch,0\n040401-20110728084714Th,20110728084714Th,20110727063234Lh,1\n040401-20111025080614Th,20111025080614Th,20111025080614Th,2\n040401-20120414115434Ch,20120414115434Ch,20120414115434Ch,0\n040401-20120421063154Uh,20120421063154Uh,20120421063154Uh,0\n040401-20120627162114Mh,20120627162114Mh,20120627162114Mh,1\n040401-20120803164614Mh,20120803164614Mh,20120802090314Th,0\n040401-20121018063134Lh,20121018063134Lh,20121018063134Lh,2\n040401-20121021145454Bh,20121021145454Bh,20121021145454Bh,1\n040401-20130824060834Lh,20130824060834Lh,20130824060834Lh,2\n040401-20130901073614Th,20130901073614Th,20130901073614Th,2\n040401-20131019080114Th,20131019080114Th,20131019080114Th,0\n040401-20131108045254Uh,20131108045254Uh,20131107043054Uh,1\n040401-20131115185414Mh,20131115185414Mh,20131115185414Mh,0\n040401-20140116180214Mh,20140116180214Mh,20140116180214Mh,0\n040401-20140522070614Th,20140522070614Th,20140522070614Th,1\n040401-20140709195814Mh,20140709195814Mh,20140709195814Mh,2\n040401-20140716195854Bh,20140716195854Bh,20140716195854Bh,1\n040401-20141010195834Ch,20141010195834Ch,20141009023454Uh,2\n040401-20150314081954Uh,20150314081954Uh,20150314081954Uh,2\n040401-20150324145954Bh,20150324145954Bh,20150323172814Mh,2\n040401-20150627133754Bh,20150627133754Bh,20150625082014Th,0\n040401-20150629161614Mh,20150629161614Mh,20150629161614Mh,2\n040401-20150830084134Lh,20150830084134Lh,20150830084134Lh,0\n040401-20150916055454Uh,20150916055454Uh,20150916055454Uh,1\n040401-20151107133734Ch,20151107133734Ch,20151105133734Ch,0\n040401-20151128133714Th,20151128133714Th,20151128133714Th,1\n040401-20151219133734Ch,20151219133734Ch,20151219133734Ch,0\n040401-20160327170714Mh,20160327170714Mh,20160326143654Bh,2\n040401-20160619133714Th,20160619133714Th,20160619133714Th,0\n040401-20160629133714Th,20160629133714Th,20160627133734Ch,2\n040401-20160927133734Ch,20160927133734Ch,20160927133734Ch,0\n040401-20170310173450Bh,20170310173450Bh,20170310133730Ch,1\n040401-20170805133710Th,20170805133710Th,20170804140650Bh,1\n040401-20170806133710Th,20170806133710Th,20170804140650Bh,1\n040401-20171115222230Lh,20171115222230Lh,20171114133730Ch,2\n040401-20180712123630Ch,20180712123630Ch,20180712123630Ch,0\n040401-20180714000010Mh,20180714000010Mh,20180714000010Mh,1\n040401-20210616142650Bh,20210616142650Bh,20210616142650Bh,2\n040401-20211119103130Ch,20211119103130Ch,20211119103130Ch,1\n040401-20220618185332Bh,20220618185332Bh,20220617185352Mh,2\n040402-20171115222230Lh,20171115222230Lh,20171114133730Ch,2\n040402-20190805054730Lh,20190805054730Lh,20190805054730Lh,1\n040402-20210616142650Bh,20210616142650Bh,20210616142650Bh,2\n050101-20110322005814Mh,20110322005814Mh,20110322005814Mh,1\n050101-20110429135554Bh,20110429135554Bh,20110429135554Bh,2\n050101-20110502063154Uh,20110502063154Uh,20110502063154Uh,0\n050101-20110517164514Mh,20110517164514Mh,20110517164514Mh,2\n050101-20110727063234Lh,20110727063234Lh,20110727063234Lh,1\n050101-20110826074214Th,20110826074214Th,20110826074214Th,1\n050101-20111114082314Th,20111114082314Th,20111114082314Th,2\n050101-20111116063134Lh,20111116063134Lh,20111114082314Th,2\n050101-20111218085014Th,20111218085014Th,20111218085014Th,2\n050101-20120110155754Bh,20120110155754Bh,20120110155754Bh,2\n050101-20120715123434Ch,20120715123434Ch,20120715123434Ch,0\n050101-20120720085114Th,20120720085114Th,20120720085114Th,1\n050101-20120810121634Ch,20120810121634Ch,20120810121634Ch,0\n050101-20120811121434Ch,20120811121434Ch,20120810121634Ch,0\n050101-20120816072814Th,20120816072814Th,20120816072814Th,0\n050101-20121007131134Ch,20121007131134Ch,20121006063534Lh,0\n050101-20121027165114Mh,20121027165114Mh,20121026063134Lh,2\n050101-20130107174854Bh,20130107174854Bh,20130107174854Bh,2\n050101-20130218220314Mh,20130218220314Mh,20130218220314Mh,0\n050101-20130305112734Ch,20130305112734Ch,20130303063134Lh,0\n050101-20130630070714Th,20130630070714Th,20130630070714Th,2\n050101-20130719140154Bh,20130719140154Bh,20130719140154Bh,0\n050101-20131225065154Uh,20131225065154Uh,20131224192154Bh,0\n050101-20131229112414Th,20131229112414Th,20131229112414Th,0\n050101-20140210180154Bh,20140210180154Bh,20140210180154Bh,2\n050101-20140414195834Ch,20140414195834Ch,20140413195914Mh,2\n050101-20140710083214Th,20140710083214Th,20140709195814Mh,2\n050101-20141214201834Ch,20141214201834Ch,20141214201834Ch,0\n050101-20141216222634Lh,20141216222634Lh,20141216222634Lh,0\n050101-20150113105034Ch,20150113105034Ch,20150112081954Uh,0\n050101-20150116082034Lh,20150116082034Lh,20150116082034Lh,1\n050101-20150216171714Mh,20150216171714Mh,20150216171714Mh,2\n050101-20150421140454Bh,20150421140454Bh,20150421140454Bh,0\n050101-20150731082034Lh,20150731082034Lh,20150731082034Lh,0\n050101-20160101133714Th,20160101133714Th,20160101133714Th,1\n050101-20160310133734Ch,20160310133734Ch,20160309172214Mh,2\n050101-20160617133654Bh,20160617133654Bh,20160617133654Bh,1\n050101-20170117175014Mh,20170117175014Mh,20170117175014Mh,0\n050101-20170609133710Th,20170609133710Th,20170609133710Th,1\n050101-20170728150230Ch,20170728150230Ch,20170728150230Ch,1\n050101-20170827165310Mh,20170827165310Mh,20170827165310Mh,0\n050101-20171115170950Bh,20171115170950Bh,20171114133730Ch,2\n050101-20171123222130Lh,20171123222130Lh,20171123133730Ch,0\n050101-20180202130330Ch,20180202130330Ch,20180202130330Ch,2\n050101-20181101230010Mh,20181101230010Mh,20181101230010Mh,0\n050101-20220328143550Bh,20220328143550Bh,20220328143550Bh,0\n050101-20220622205632Bh,20220622205632Bh,20220622205632Bh,2\n050102-20110322005814Mh,20110322005814Mh,20110322005814Mh,1\n050102-20110429135554Bh,20110429135554Bh,20110429135554Bh,2\n050102-20110502063154Uh,20110502063154Uh,20110502063154Uh,0\n050102-20110517164514Mh,20110517164514Mh,20110517164514Mh,2\n050102-20110727063234Lh,20110727063234Lh,20110727063234Lh,1\n050102-20110826074214Th,20110826074214Th,20110826074214Th,1\n050102-20111114082314Th,20111114082314Th,20111114082314Th,2\n050102-20111116063134Lh,20111116063134Lh,20111114082314Th,2\n050102-20111218085014Th,20111218085014Th,20111218085014Th,2\n050102-20120110155754Bh,20120110155754Bh,20120110155754Bh,2\n050102-20120715123434Ch,20120715123434Ch,20120715123434Ch,0\n050102-20120720085114Th,20120720085114Th,20120720085114Th,1\n050102-20120811121434Ch,20120811121434Ch,20120810121634Ch,0\n050102-20120816072814Th,20120816072814Th,20120816072814Th,0\n050102-20121027165114Mh,20121027165114Mh,20121026063134Lh,2\n050102-20130107174854Bh,20130107174854Bh,20130107174854Bh,2\n050102-20130218220314Mh,20130218220314Mh,20130218220314Mh,0\n050102-20130305112734Ch,20130305112734Ch,20130303063134Lh,0\n050102-20130630070714Th,20130630070714Th,20130630070714Th,2\n050102-20130719140154Bh,20130719140154Bh,20130719140154Bh,0\n050102-20131225065154Uh,20131225065154Uh,20131224192154Bh,0\n050102-20131229112414Th,20131229112414Th,20131229112414Th,0\n050102-20140210180154Bh,20140210180154Bh,20140210180154Bh,2\n050102-20140414195834Ch,20140414195834Ch,20140413195914Mh,2\n050102-20140710083214Th,20140710083214Th,20140709195814Mh,2\n050102-20141214201834Ch,20141214201834Ch,20141214201834Ch,0\n050102-20141216222634Lh,20141216222634Lh,20141216222634Lh,0\n050102-20150113105034Ch,20150113105034Ch,20150112081954Uh,0\n050102-20150116082034Lh,20150116082034Lh,20150116082034Lh,1\n050102-20150216171714Mh,20150216171714Mh,20150216171714Mh,2\n050102-20150421140454Bh,20150421140454Bh,20150421140454Bh,0\n050102-20150731082034Lh,20150731082034Lh,20150731082034Lh,0\n050102-20160101133714Th,20160101133714Th,20160101133714Th,1\n050102-20160310133734Ch,20160310133734Ch,20160309172214Mh,2\n050102-20160617133654Bh,20160617133654Bh,20160617133654Bh,1\n050102-20170117175014Mh,20170117175014Mh,20170117175014Mh,0\n050102-20170609133710Th,20170609133710Th,20170609133710Th,1\n050102-20170728150230Ch,20170728150230Ch,20170728150230Ch,1\n050102-20170827165310Mh,20170827165310Mh,20170827165310Mh,0\n050102-20171123222130Lh,20171123222130Lh,20171123133730Ch,0\n050102-20180202130330Ch,20180202130330Ch,20180202130330Ch,2\n050102-20220328143550Bh,20220328143550Bh,20220328143550Bh,0\n050102-20220622205632Bh,20220622205632Bh,20220622205632Bh,2\n050103-20110322005814Mh,20110322005814Mh,20110322005814Mh,1\n050103-20110429135554Bh,20110429135554Bh,20110429135554Bh,2\n050103-20110502063154Uh,20110502063154Uh,20110502063154Uh,0\n050103-20110517164514Mh,20110517164514Mh,20110517164514Mh,2\n050103-20110727063234Lh,20110727063234Lh,20110727063234Lh,1\n050103-20110826074214Th,20110826074214Th,20110826074214Th,1\n050103-20111114082314Th,20111114082314Th,20111114082314Th,2\n050103-20111116063134Lh,20111116063134Lh,20111114082314Th,2\n050103-20111218085014Th,20111218085014Th,20111218085014Th,2\n050103-20120110155754Bh,20120110155754Bh,20120110155754Bh,2\n050103-20120715123434Ch,20120715123434Ch,20120715123434Ch,0\n050103-20120720085114Th,20120720085114Th,20120720085114Th,1\n050103-20120810121634Ch,20120810121634Ch,20120810121634Ch,0\n050103-20120811121434Ch,20120811121434Ch,20120810121634Ch,0\n050103-20120816072814Th,20120816072814Th,20120816072814Th,0\n050103-20121007131134Ch,20121007131134Ch,20121006063534Lh,0\n050103-20121027165114Mh,20121027165114Mh,20121026063134Lh,2\n050103-20130107174854Bh,20130107174854Bh,20130107174854Bh,2\n050103-20130218220314Mh,20130218220314Mh,20130218220314Mh,0\n050103-20130305112734Ch,20130305112734Ch,20130303063134Lh,0\n050103-20130630070714Th,20130630070714Th,20130630070714Th,2\n050103-20130719140154Bh,20130719140154Bh,20130719140154Bh,0\n050103-20131225065154Uh,20131225065154Uh,20131224192154Bh,0\n050103-20131229112414Th,20131229112414Th,20131229112414Th,0\n050103-20140210180154Bh,20140210180154Bh,20140210180154Bh,2\n050103-20140414195834Ch,20140414195834Ch,20140413195914Mh,2\n050103-20140710083214Th,20140710083214Th,20140709195814Mh,2\n050103-20141214201834Ch,20141214201834Ch,20141214201834Ch,0\n050103-20141216222634Lh,20141216222634Lh,20141216222634Lh,0\n050103-20150113105034Ch,20150113105034Ch,20150112081954Uh,0\n050103-20150116082034Lh,20150116082034Lh,20150116082034Lh,1\n050103-20150216171714Mh,20150216171714Mh,20150216171714Mh,2\n050103-20150421140454Bh,20150421140454Bh,20150421140454Bh,0\n050103-20150731082034Lh,20150731082034Lh,20150731082034Lh,0\n050103-20160101133714Th,20160101133714Th,20160101133714Th,1\n050103-20160310133734Ch,20160310133734Ch,20160309172214Mh,2\n050103-20160617133654Bh,20160617133654Bh,20160617133654Bh,1\n050103-20170117175014Mh,20170117175014Mh,20170117175014Mh,0\n050103-20170609133710Th,20170609133710Th,20170609133710Th,1\n050103-20170728150230Ch,20170728150230Ch,20170728150230Ch,1\n050103-20170827165310Mh,20170827165310Mh,20170827165310Mh,0\n050103-20171115170950Bh,20171115170950Bh,20171114133730Ch,2\n050103-20171123222130Lh,20171123222130Lh,20171123133730Ch,0\n050103-20180202130330Ch,20180202130330Ch,20180202130330Ch,2\n050103-20180801232450Bh,20180801232450Bh,20180801232450Bh,0\n050103-20181101230010Mh,20181101230010Mh,20181101230010Mh,0\n050103-20220328143550Bh,20220328143550Bh,20220328143550Bh,0\n050103-20220622205632Bh,20220622205632Bh,20220622205632Bh,2\n050201-20160928143554Bh,20160928143554Bh,20160927133734Ch,0\n050202-20110425063154Uh,20110425063154Uh,20110425063154Uh,0\n050202-20120127085214Th,20120127085214Th,20120127085214Th,0\n050202-20120213063134Lh,20120213063134Lh,20120213063134Lh,0\n050202-20120523163014Mh,20120523163014Mh,20120523163014Mh,0\n050202-20120730135454Bh,20120730135454Bh,20120729122634Ch,2\n050202-20121219063154Uh,20121219063154Uh,20121218063534Lh,2\n050202-20130616070414Th,20130616070414Th,20130615002434Lh,1\n050202-20131013185334Ch,20131013185334Ch,20131013185334Ch,1\n050202-20140128174554Bh,20140128174554Bh,20140127180134Ch,2\n050202-20140803234534Lh,20140803234534Lh,20140802072114Th,1\n050202-20140908231334Lh,20140908231334Lh,20140908175354Bh,2\n050202-20150327145114Th,20150327145114Th,20150327145114Th,2\n050202-20150621161314Mh,20150621161314Mh,20150620082014Th,2\n050202-20150913082014Th,20150913082014Th,20150913082014Th,1\n050202-20150926111934Ch,20150926111934Ch,20150926111934Ch,2\n050202-20151116224034Lh,20151116224034Lh,20151116201114Mh,1\n050202-20160326143654Bh,20160326143654Bh,20160326143654Bh,2\n050202-20160430133734Ch,20160430133734Ch,20160430133714Th,0\n050202-20160720133734Ch,20160720133734Ch,20160720133734Ch,1\n050202-20160928143554Bh,20160928143554Bh,20160927133734Ch,0\n050202-20161229133734Ch,20161229133734Ch,20161229133734Ch,2\n050202-20170322015550Uh,20170322015550Uh,20170322015550Uh,0\n050202-20171005144050Bh,20171005144050Bh,20171004143950Bh,2\n050302-20110211084114Th,20110211084114Th,20110211084114Th,0\n050302-20120113104834Ch,20120113104834Ch,20120113104834Ch,0\n050302-20130808140054Bh,20130808140054Bh,20130807073114Th,2\n050302-20170102024454Uh,20170102024454Uh,20170102024454Uh,0\n050302-20170503013050Uh,20170503013050Uh,20170502233130Lh,1\n050302-20181102104630Ch,20181102104630Ch,20181101230010Mh,0\n050302-20190317145150Bh,20190317145150Bh,20190317145150Bh,0\n050302-20190926165810Mh,20190926165810Mh,20190925165810Mh,2\n050303-20110810063134Lh,20110810063134Lh,20110810063134Lh,2\n050303-20120113104834Ch,20120113104834Ch,20120113104834Ch,0\n050303-20121208171514Mh,20121208171514Mh,20121207103034Ch,0\n050303-20141001195834Ch,20141001195834Ch,20141001195834Ch,1\n050303-20150305081954Uh,20150305081954Uh,20150305081954Uh,1\n050303-20150620082014Th,20150620082014Th,20150620082014Th,2\n050303-20151105133734Ch,20151105133734Ch,20151105133734Ch,0\n050303-20151209201754Bh,20151209201754Bh,20151209201754Bh,0\n050303-20160925225634Lh,20160925225634Lh,20160925165814Mh,1\n050303-20190926165810Mh,20190926165810Mh,20190925165810Mh,2\n060102-20110326170914Mh,20110326170914Mh,20110324082614Th,2\n060102-20130222082914Th,20130222082914Th,20130221063134Lh,2\n060102-20130419070254Uh,20130419070254Uh,20130419070254Uh,1\n060102-20130423072414Th,20130423072414Th,20130423072414Th,0\n060102-20140122180214Mh,20140122180214Mh,20140122180214Mh,2\n060102-20140818195814Mh,20140818195814Mh,20140816072814Th,1\n060102-20150207084414Th,20150207084414Th,20150207084414Th,1\n060102-20150323172814Mh,20150323172814Mh,20150323172814Mh,2\n060102-20151019223534Lh,20151019223534Lh,20151019223534Lh,0\n060102-20151228024254Uh,20151228024254Uh,20151228024254Uh,2\n060102-20160724135554Bh,20160724135554Bh,20160724135554Bh,0\n060102-20160809164914Mh,20160809164914Mh,20160808133714Th,0\n060102-20161023133814Th,20161023133814Th,20161023133814Th,2\n060102-20170217165250Bh,20170217165250Bh,20170217165250Bh,0\n060102-20170316133730Ch,20170316133730Ch,20170316133730Ch,1\n060102-20170809164910Mh,20170809164910Mh,20170809164910Mh,1\n060102-20180827232530Lh,20180827232530Lh,20180827232530Lh,1\n060102-20180924112030Ch,20180924112030Ch,20180923112330Ch,0\n060102-20190322144450Bh,20190322144450Bh,20190322144450Bh,0\n060102-20201228202430Ch,20201228202430Ch,20201227202710Mh,1\n060102-20210329114430Ch,20210329114430Ch,20210329015250Uh,0\n060102-20210721123430Ch,20210721123430Ch,20210721123430Ch,1\n060102-20220724185352Mh,20220724185352Mh,20220724185352Mh,2\n060403-20110123174914Mh,20110123174914Mh,20110123174914Mh,2\n060403-20110221173814Mh,20110221173814Mh,20110220082634Lh,2\n060403-20110313082654Uh,20110313082654Uh,20110312082634Lh,2\n060403-20110523063154Uh,20110523063154Uh,20110523063154Uh,2\n060403-20110615133454Bh,20110615133454Bh,20110615133454Bh,1\n060403-20110621063134Lh,20110621063134Lh,20110621063134Lh,2\n060403-20111226155454Bh,20111226155454Bh,20111226155454Bh,0\n060403-20120114063154Uh,20120114063154Uh,20120113104834Ch,0\n060403-20120509163414Mh,20120509163414Mh,20120509163414Mh,1\n060403-20120813164714Mh,20120813164714Mh,20120813164714Mh,1\n060403-20120814140454Bh,20120814140454Bh,20120813164714Mh,1\n060403-20120829063134Lh,20120829063134Lh,20120829063134Lh,0\n060403-20121030150354Bh,20121030150354Bh,20121030150354Bh,1\n060403-20130212153254Bh,20130212153254Bh,20130211063134Lh,0\n060403-20130327175114Mh,20130327175114Mh,20130326063134Lh,0\n060403-20130411115134Ch,20130411115134Ch,20130411115134Ch,1\n060403-20130601063154Uh,20130601063154Uh,20130601063154Uh,1\n060403-20130628063134Lh,20130628063134Lh,20130628063134Lh,1\n060403-20130707134154Bh,20130707134154Bh,20130707134154Bh,2\n060403-20131028190734Ch,20131028190734Ch,20131027025654Uh,2\n060403-20131207084214Th,20131207084214Th,20131206084114Th,0\n060403-20140411013854Uh,20140411013854Uh,20140411013854Uh,2\n060403-20140705035754Uh,20140705035754Uh,20140705035754Uh,0\n060403-20140814195854Bh,20140814195854Bh,20140813064254Uh,2\n060403-20140927014554Uh,20140927014554Uh,20140927014554Uh,1\n060403-20141012195854Bh,20141012195854Bh,20141012015654Uh,1\n060403-20141026024454Uh,20141026024454Uh,20141024195834Ch,2\n060403-20141225172214Mh,20141225172214Mh,20141225024354Uh,1\n060403-20150113155654Bh,20150113155654Bh,20150112081954Uh,0\n060403-20150409081954Uh,20150409081954Uh,20150409081954Uh,0\n060403-20150417082314Th,20150417082314Th,20150417082314Th,1\n060403-20150527082014Th,20150527082014Th,20150527082014Th,2\n060403-20151116201114Mh,20151116201114Mh,20151116201114Mh,1\n060403-20160107133714Th,20160107133714Th,20160107133714Th,2\n060403-20160110133714Th,20160110133714Th,20160110133714Th,0\n060403-20160530234734Lh,20160530234734Lh,20160530234734Lh,1\n060403-20160804133734Ch,20160804133734Ch,20160804133734Ch,0\n060403-20160925165814Mh,20160925165814Mh,20160925165814Mh,1\n060403-20170125133734Ch,20170125133734Ch,20170125133734Ch,0\n060403-20180301050430Lh,20180301050430Lh,20180301050430Lh,2\n060403-20180923112330Ch,20180923112330Ch,20180923112330Ch,0\n060403-20181014105830Ch,20181014105830Ch,20181014105830Ch,0\n060403-20211222120130Ch,20211222120130Ch,20211222120130Ch,1\n'}  # filled by code/build_e004_kernel.py: {filename: source text}
EMBEDDED_FOLDS_SHA256 = '6316d1e5cd628f01b0073ab12caea91c38533a4e31366e2a1360a6863115b6ec'
SMP_VERSION = '0.5.0'
SEED = 20261027
GRID_T = (0.3, 0.4, 0.5, 0.6)
GRID_MIN_AREA = (0, 64, 256)
GRID_DIL = (0, 5)
DEFAULT_SETTING = (0.5, 0, 0)  # threshold, merge-dilation radius, min_area
OUT = None
ENCODER = 'resnet34'


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    return h.hexdigest()


def elapsed():
    return time.monotonic() - T0


def log(msg):
    line = f'[{elapsed():8.1f}s] {msg}'
    print(line, flush=True)
    if OUT is not None:
        with open(OUT / 'train.log', 'a', encoding='utf-8') as f:
            f.write(line + '\n')


def dump(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str, allow_nan=True), encoding='utf-8')
    tmp.replace(path)


def setup_lib(out):
    """Embedded helper modules (pq_metric, validate_submission, ...) become importable."""
    if EMBEDDED:
        lib = out / 'lib'
        lib.mkdir(parents=True, exist_ok=True)
        for name, text in EMBEDDED.items():
            (lib / name).write_text(text, encoding='utf-8')
        sys.path.insert(0, str(lib))
        return {n: sha_bytes(t.encode('utf-8')) for n, t in EMBEDDED.items()}
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    return {}


# ------------------------------------------------------------------ environment
def pip_freeze():
    r = subprocess.run([sys.executable, '-m', 'pip', 'freeze'], capture_output=True, text=True)
    return dict(l.split('==', 1) for l in r.stdout.splitlines() if '==' in l)


def ensure_smp(args, receipt):
    info = {'pinned': SMP_VERSION}
    try:
        info['preinstalled'] = importlib.metadata.version('segmentation_models_pytorch')
    except importlib.metadata.PackageNotFoundError:
        info['preinstalled'] = None
    if not args.skip_pip and info['preinstalled'] != SMP_VERSION:
        before = pip_freeze()
        r = subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-input', '--disable-pip-version-check',
                            f'segmentation-models-pytorch=={SMP_VERSION}'], capture_output=True, text=True)
        info['pip_install_returncode'] = r.returncode
        info['pip_install_tail'] = r.stdout[-1500:] + r.stderr[-1500:]
        if r.returncode:
            raise RuntimeError('pip install of pinned smp failed')
        after = pip_freeze()
        info['pip_changed'] = {k: [before.get(k), v] for k, v in after.items() if before.get(k) != v}
        importlib.invalidate_caches()
    import segmentation_models_pytorch as smp
    info['version'] = smp.__version__
    if not args.skip_pip and smp.__version__ != SMP_VERSION:
        raise RuntimeError(f'smp {smp.__version__} != pinned {SMP_VERSION}')
    # Wheels for offline E005: smp + its non-image-core requirements at installed versions.
    if args.wheels:
        wheels = OUT / 'wheels'
        wheels.mkdir(exist_ok=True)
        reqs = {'segmentation-models-pytorch': smp.__version__}
        skip = {'torch', 'torchvision', 'numpy', 'pillow', 'tqdm'}
        for req in importlib.metadata.requires('segmentation_models_pytorch') or []:
            name = req.split(';')[0].split('>')[0].split('<')[0].split('=')[0].split('!')[0].split('[')[0].strip()
            if 'extra ==' in req or name.lower() in skip:
                continue
            try:
                reqs[name] = importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                pass
        for k, v in (info.get('pip_changed') or {}).items():
            if v[1] and k.lower() not in skip:
                reqs[k] = v[1]
        dl = {}
        for name, ver in sorted(reqs.items()):
            r = subprocess.run([sys.executable, '-m', 'pip', 'download', '--no-deps', '--no-input',
                                '--disable-pip-version-check', '-d', str(wheels), f'{name}=={ver}'],
                               capture_output=True, text=True)
            dl[f'{name}=={ver}'] = r.returncode
        info['wheel_download_returncodes'] = dl
        info['wheels'] = {p.name: sha_file(p) for p in sorted(wheels.iterdir())}
    receipt['smp'] = info
    return smp


def versions():
    out = {}
    for p in ('torch', 'torchvision', 'segmentation_models_pytorch', 'timm', 'huggingface_hub', 'safetensors',
              'numpy', 'Pillow', 'pycocotools', 'scipy'):
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


def load_records(ann_path, images):
    data = json.loads(ann_path.read_text(encoding='utf-8'))
    by_name = {p.name: s for s, (split, p) in images.items() if split == 'train'}
    rec_phys = {}
    for r in data['images']:
        name = Path(r['file_name']).name
        if name not in by_name:
            raise ValueError(f'record {r["id"]} has unresolved file {name}')
        rec_phys[str(r['id'])] = by_name[name]
    segs = {}
    for a in data['annotations']:
        segs.setdefault(str(a['image_id']), []).append(a['segmentation'])
    return rec_phys, segs


def decode_seg(seg, size=2048):
    from pycocotools import mask as cm
    if isinstance(seg, list):
        rle = cm.merge(cm.frPyObjects(seg, size, size))
    else:
        rle = cm.frPyObjects(seg, size, size) if isinstance(seg.get('counts'), list) else \
            {'size': seg['size'], 'counts': seg['counts'].encode('ascii') if isinstance(seg['counts'], str) else seg['counts']}
    return cm.decode(rle)


def prep_image(task):
    """Worker: returns (image uint8 SxS, soft target uint8 SxS, gt crops per record or None)."""
    import numpy as np
    from PIL import Image
    path, record_segs, size, want_gt = task
    with Image.open(path) as im:
        g = im.convert('L')
        if g.size != (2048, 2048):
            raise ValueError(f'{path}: size {g.size}')
        f = 2048 // size
        small = g.reduce(f) if 2048 % size == 0 else g.resize((size, size), Image.Resampling.BOX)
        img = np.asarray(small, dtype=np.uint8).copy()
    target = np.zeros((size, size), np.float32)
    gts = [] if want_gt else None
    for segs in record_segs:
        union = np.zeros((2048, 2048), np.uint8)
        crops = []
        for seg in segs:
            m = decode_seg(seg)
            union |= m
            if want_gt:
                ys, xs = np.nonzero(m)
                if len(ys) == 0:
                    crops.append((0, 0, np.zeros((0, 0), bool), 0))  # keeps official stack count
                    continue
                y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
                crops.append((int(y0), int(x0), m[y0:y1, x0:x1].astype(bool), int(m.sum())))
        if 2048 % size == 0:
            target += union.reshape(size, f, size, f).mean(axis=(1, 3))
        else:
            target += np.asarray(Image.fromarray(union * 255).resize((size, size), Image.Resampling.BOX), np.float32) / 255
        if want_gt:
            gts.append(crops)
    target /= max(1, len(record_segs))
    return img, np.round(target * 255).astype(np.uint8), gts


def run_pool(fn, tasks, workers):
    if workers <= 1:
        return [fn(t) for t in tasks]
    import multiprocessing as mp
    with mp.get_context('fork' if sys.platform != 'win32' else 'spawn').Pool(workers) as pool:
        return pool.map(fn, tasks, chunksize=4)


def read_folds(args):
    if args.folds_csv:
        text = Path(args.folds_csv).read_text(encoding='utf-8')
    else:
        text = EMBEDDED['folds.csv']
        if EMBEDDED_FOLDS_SHA256 and sha_bytes(text.encode('utf-8')) != EMBEDDED_FOLDS_SHA256:
            raise ValueError('embedded folds.csv hash mismatch')
    rows = list(csv.DictReader(text.splitlines()))
    return {r['record_id']: (r['physical_id'], int(r['fold'])) for r in rows}, sha_bytes(text.encode('utf-8'))


# ------------------------------------------------------------------ model / training
def build_model(smp, weights):
    return smp.Unet(encoder_name=ENCODER, encoder_weights=weights, in_channels=1, classes=1)


def normalize(x):
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    s = x.std(dim=(1, 2, 3), keepdim=True)
    return (x - m) / (s + 1e-6)


def augment(x, y, rng):
    import torch
    xs, ys = [], []
    for i in range(x.shape[0]):
        xi, yi = x[i], y[i]
        k = rng.randrange(4)
        if k:
            xi, yi = torch.rot90(xi, k, (1, 2)), torch.rot90(yi, k, (1, 2))
        if rng.random() < 0.5:
            xi, yi = xi.flip(2), yi.flip(2)
        gamma = math.exp(rng.uniform(math.log(0.8), math.log(1.25)))
        c, b = rng.uniform(0.9, 1.1), rng.uniform(-0.05, 0.05)
        xi = xi.clamp(1e-4, 1) ** gamma
        mu = xi.mean()
        xi = ((xi - mu) * c + mu + b).clamp(0, 1)
        xs.append(xi)
        ys.append(yi)
    return torch.stack(xs), torch.stack(ys)


def loss_fn(logits, y):
    import torch.nn.functional as F
    logits = logits.float()
    bce = F.binary_cross_entropy_with_logits(logits, y)
    p = logits.sigmoid()
    inter = (p * y).sum(dim=(1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum(dim=(1, 2, 3)) + y.sum(dim=(1, 2, 3)) + 1)
    return bce + dice.mean(), bce.item(), dice.mean().item()


def predict(model, imgs, device, batch, amp):
    """imgs: uint8 tensor [N,1,S,S] -> float16 numpy probs [N,S,S]."""
    import numpy as np
    import torch
    model.eval()
    out = np.zeros((imgs.shape[0], imgs.shape[2], imgs.shape[3]), np.float16)
    with torch.no_grad():
        for i in range(0, imgs.shape[0], batch):
            x = normalize(imgs[i:i + batch].to(device).float() / 255)
            if device.type == 'cuda':
                x = x.contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(x)
            out[i:i + batch] = logits.float().sigmoid()[:, 0].cpu().numpy().astype(np.float16)
    return out


# ------------------------------------------------------------------ post-processing / PQ
def dilate(binary, radius, device):
    import numpy as np
    import torch
    import torch.nn.functional as F
    yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
    disk = (yy ** 2 + xx ** 2 <= radius ** 2)
    if device.type == 'cuda':
        k = torch.tensor(disk, dtype=torch.float16, device=device)[None, None]
        b = torch.from_numpy(binary).to(device).half()[None, None]
        return (F.conv2d(b, k, padding=radius)[0, 0] > 0.5).cpu().numpy()
    from scipy import ndimage
    return ndimage.binary_dilation(binary, structure=disk)


def label_instances(binary, radius, device):
    """8-connected components; with radius>0 components are found on the dilated mask and
    their labels applied back to the undilated mask."""
    import numpy as np
    from scipy import ndimage
    st = np.ones((3, 3), bool)
    if radius:
        lab, n = ndimage.label(dilate(binary, radius, device), structure=st)
        lab[~binary] = 0
    else:
        lab, n = ndimage.label(binary, structure=st)
    return lab.astype(np.int32), n


def filter_min_area(lab, n, min_area):
    import numpy as np
    areas = np.bincount(lab.ravel(), minlength=n + 1)
    keep = areas >= max(1, min_area)
    keep[0] = False
    remap = np.zeros(n + 1, np.int32)
    remap[keep] = np.arange(1, int(keep.sum()) + 1, dtype=np.int32)
    return remap[lab], int(keep.sum())


def fast_overlap(gt_crops, lab, n):
    """Same float32 IoU matrix as pq_metric.overlap_matrix for (GT stack, label-map instances)."""
    import numpy as np
    areas = np.bincount(lab.ravel(), minlength=n + 1).astype(np.int64)
    inter = np.zeros((len(gt_crops), n), np.int64)
    gt_area = np.zeros(len(gt_crops), np.int64)
    for i, (y0, x0, crop, a) in enumerate(gt_crops):
        gt_area[i] = a
        if a and n:
            sub = lab[y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]][crop]
            inter[i] = np.bincount(sub, minlength=n + 1)[1:n + 1]
    i32 = inter.astype(np.float32)
    union = gt_area.astype(np.float32)[:, None] + areas[1:].astype(np.float32)[None, :] - i32
    return np.divide(i32, union, out=np.zeros_like(i32), where=union != 0), inter, gt_area


def dense_stack(gt_crops, size=2048):
    import numpy as np
    s = np.zeros((len(gt_crops), size, size), np.uint8)
    for i, (y0, x0, crop, a) in enumerate(gt_crops):
        if a:
            s[i, y0:y0 + crop.shape[0], x0:x0 + crop.shape[1]] = crop
    return s


def upsample(prob16, device):
    import torch
    import torch.nn.functional as F
    t = torch.from_numpy(prob16.astype('float32')).to(device)[None, None]
    return F.interpolate(t, size=(2048, 2048), mode='bilinear', align_corners=False)[0, 0]


def frag_merge(inter, gt_area, frac=0.1):
    import numpy as np
    if inter.size == 0:
        return 0, 0
    sig = inter >= np.maximum(1, frac * gt_area[:, None])
    return int((sig.sum(axis=1) >= 2).sum()), int((sig.sum(axis=0) >= 2).sum())


def score_settings(settings, oof_ids, probs, entries_by_img, device, deadline, pq, parity_images=0):
    """settings: list of (t, radius, min_area). Returns per-setting results and run info."""
    import numpy as np
    by_tr = {}
    for t, r, a in settings:
        by_tr.setdefault((t, r), []).append(a)
    mats = {s: [] for s in settings}
    extra = {s: {'pred_instances': 0, 'fragmented_gt': 0, 'merged_pred': 0, 'images_with_preds': 0} for s in settings}
    parity = []
    done, t_first = 0, None
    for i, pid in enumerate(oof_ids):
        if elapsed() > deadline:
            break
        ts = time.monotonic()
        up = upsample(probs[i], device)
        for (t, r), min_areas in by_tr.items():
            binary = (up > t).cpu().numpy()
            lab0, n0 = label_instances(binary, r, device)
            for a in min_areas:
                lab, n = filter_min_area(lab0, n0, a)
                s = (t, r, a)
                extra[s]['pred_instances'] += n
                extra[s]['images_with_preds'] += n > 0
                for crops in entries_by_img[pid]:
                    m, inter, gta = fast_overlap(crops, lab, n)
                    mats[s].append(m)
                    fg, mg = frag_merge(inter, gta)
                    extra[s]['fragmented_gt'] += fg
                    extra[s]['merged_pred'] += mg
                    if i < parity_images and s == DEFAULT_SETTING and n <= 40 and len(crops) <= 30:
                        pred = np.stack([(lab == k).astype(np.uint8) for k in range(1, n + 1)]) if n else \
                            np.zeros((0, 2048, 2048), np.uint8)
                        ref = pq.overlap_matrix(dense_stack(crops), pred)
                        parity.append({'image': pid, 'equal': bool(np.array_equal(ref, m)), 'shape': list(m.shape)})
        done += 1
        if t_first is None:
            t_first = time.monotonic() - ts
    results = {}
    for s in settings:
        res = pq.score_matrices(mats[s])
        res.update(extra[s])
        res['images_scored'] = done
        results[s] = res
    return results, {'images_scored': done, 'images_total': len(oof_ids), 'first_image_seconds': t_first,
                     'complete': done == len(oof_ids), 'parity': parity}


# ------------------------------------------------------------------ round-trip (fresh process)
def roundtrip(args):
    """Fresh process: rebuild model without pretrained weights, load checkpoint, predict a few
    held-out images, post-process, RLE-encode, validate, decode back."""
    import numpy as np
    import torch
    from pycocotools import mask as cm
    sys.path.insert(0, str(Path(args.lib)))
    import validate_submission as vs
    import segmentation_models_pytorch as smp
    rt = json.loads(Path(args.roundtrip).read_text(encoding='utf-8'))
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    ck = torch.load(rt['checkpoint'], map_location='cpu', weights_only=False)
    model = build_model(smp, None)
    missing = model.load_state_dict(ck['state_dict'], strict=True)
    model.to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    oof = np.load(rt['oof_path'], mmap_mode='r')
    oof_ids = json.loads(Path(rt['oof_ids_path']).read_text(encoding='utf-8'))
    _, images = discover(Path(rt['input']))
    from PIL import Image
    size = ck['config']['size']
    rows, report = [], {'images': {}, 'strict_load': str(missing)}
    masks = {}
    for pid in rt['ids']:
        with Image.open(images[pid][1]) as im:
            g = im.convert('L')
            small = g.reduce(2048 // size) if 2048 % size == 0 else g.resize((size, size), Image.Resampling.BOX)
        x = torch.from_numpy(np.asarray(small, np.uint8).copy())[None, None]
        p = predict(model, x, device, 1, rt['amp'])[0]
        ref = oof[oof_ids.index(pid)]
        up = upsample(p, device)
        lab, n = label_instances((up > 0.5).cpu().numpy(), 0, device)
        lab, n = filter_min_area(lab, n, 0)
        for k in range(1, n + 1):
            m = np.asfortranarray((lab == k).astype(np.uint8))
            counts = cm.encode(m)['counts'].decode('ascii')
            rows.append((f'{pid}_{k}', counts))
            masks[f'{pid}_{k}'] = m
        report['images'][pid] = {'max_abs_prob_diff_vs_oof': float(np.abs(p.astype(np.float32) - ref.astype(np.float32)).max()),
                                 'instances': n, 'expected_instances_main_process': rt['expected_counts'].get(pid)}
    sub = Path(rt['out']) / 'roundtrip_submission.csv'
    with open(sub, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['filament_id', 'segmentation_rle'])
        w.writerows(rows)
    report['validator'] = vs.validate_submission(sub, allowed_ids=rt['ids'], reject_empty_masks=True, reject_all_empty=True)
    csv.field_size_limit(64 * 1024 * 1024)
    mismatch = 0
    with open(sub, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            d = cm.decode({'size': [2048, 2048], 'counts': r['segmentation_rle'].encode('ascii')})
            mismatch += not np.array_equal(d, masks[r['filament_id']])
    report['decode_mismatches'] = mismatch
    report['rows'] = len(rows)
    report['instance_counts_match_main_process'] = all(
        v['instances'] == v['expected_instances_main_process'] for v in report['images'].values())
    report['passed'] = (mismatch == 0 and report['validator']['valid'] and
                        all(v['max_abs_prob_diff_vs_oof'] < 2e-2 for v in report['images'].values()))
    dump(Path(rt['out']) / 'roundtrip.json', report)
    print(json.dumps({'roundtrip_passed': report['passed'], 'rows': len(rows)}))


# ------------------------------------------------------------------ main
def main():
    global OUT
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=Path('/kaggle/input'))
    ap.add_argument('--output', type=Path, default=Path('/kaggle/working'))
    ap.add_argument('--folds-csv', default=None)
    ap.add_argument('--fold', type=int, default=1)
    ap.add_argument('--size', type=int, default=1024)
    ap.add_argument('--epochs', type=int, default=30)
    ap.add_argument('--batch', type=int, default=8)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--warmup-steps', type=int, default=50)
    ap.add_argument('--encoder-weights', default='imagenet')
    ap.add_argument('--device', default='auto')
    ap.add_argument('--workers', type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument('--cap-seconds', type=float, default=7200.0, help='server session cap')
    ap.add_argument('--reserve-seconds', type=float, default=900)
    ap.add_argument('--startup-seconds', type=float, default=180, help='session time before script start')
    ap.add_argument('--post-seconds', type=float, default=300, help='estimate: OOF save + round-trip + default PQ')
    ap.add_argument('--val-every', type=int, default=5)
    ap.add_argument('--skip-pip', action='store_true')
    ap.add_argument('--no-wheels', dest='wheels', action='store_false')
    ap.add_argument('--require-gpu', action='store_true')
    ap.add_argument('--roundtrip', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--lib', default=None, help=argparse.SUPPRESS)
    ap.add_argument('--encoder', default='tu-convnext_tiny.fb_in22k_ft_in1k')
    ap.add_argument('--micro', type=int, default=0, help='initial micro-batch (0 = batch); also the validation batch')
    ap.add_argument('--seed', type=int, default=20261027, help='E059 (D2) uses 20261028')
    ap.add_argument('--skip-post', action='store_true', help='stop after checkpoint + single-view OOF (no PQ grid / round-trip)')
    args = ap.parse_args()
    global ENCODER, SEED
    SEED = args.seed
    ENCODER = args.encoder
    if args.roundtrip:
        return roundtrip(args)
    if EMBEDDED and args.device == 'auto':
        args.require_gpu = True  # Kaggle build: never silently train on CPU
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    receipt = {'experiment': 'E056', 'fold': args.fold, 'status': 'starting', 'source_sha256': sha_file(__file__),
               'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}}
    receipt['embedded_sha256'] = setup_lib(OUT)
    rpath = OUT / 'e004_receipt.json'
    dump(rpath, receipt)

    # Hard watchdog: never let the server timeout discard everything silently.
    hard = args.cap_seconds - args.startup_seconds - 60

    def watchdog():
        try:
            last = json.loads(rpath.read_text(encoding='utf-8'))
            last.update(status='watchdog_timeout', terminal=True)
            dump(rpath, last)
        finally:
            os._exit(124)
    timer = threading.Timer(hard, watchdog)
    timer.daemon = True
    timer.start()
    try:
        _main(args, receipt, rpath)
    except Exception as exc:
        import traceback
        receipt.update(status='exception', exception=repr(exc), traceback=traceback.format_exc()[-4000:], terminal=True)
        dump(rpath, receipt)
        log(f'EXCEPTION {exc!r}')
        raise
    finally:
        timer.cancel()


def _main(args, receipt, rpath):
    import numpy as np
    smp = ensure_smp(args, receipt)
    import torch
    import pq_metric as pq
    receipt['versions'] = versions()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    log(f'versions {receipt["versions"]}')

    # ---- data
    folds, folds_sha = read_folds(args)
    receipt['folds_csv_sha256'] = folds_sha
    ann_path, images = discover(args.input)
    rec_phys, segs = load_records(ann_path, images)
    receipt['annotation_sha256'] = sha_file(ann_path)
    if set(rec_phys) != set(folds):
        raise ValueError(f'folds/annotation record mismatch: {len(set(rec_phys) ^ set(folds))}')
    phys_fold, phys_recs = {}, {}
    for rid, pid in sorted(rec_phys.items()):
        fp, k = folds[rid]
        if fp != pid or phys_fold.setdefault(pid, k) != k:
            raise ValueError(f'fold inconsistency at {rid}')
        if rid in segs:  # official entries come from annotation rows
            phys_recs.setdefault(pid, []).append(rid)
    pids = sorted(phys_recs)
    val_ids = [p for p in pids if phys_fold[p] == args.fold]
    tr_ids = [p for p in pids if phys_fold[p] != args.fold]
    receipt['data'] = {'train_images': len(tr_ids), 'val_images': len(val_ids),
                       'val_records': sum(len(phys_recs[p]) for p in val_ids),
                       'val_instances': sum(len(segs[r]) for p in val_ids for r in phys_recs[p])}
    log(f'data {receipt["data"]}')
    t = time.monotonic()
    tasks = [(str(images[p][1]), [segs[r] for r in phys_recs[p]], args.size, phys_fold[p] == args.fold) for p in pids]
    prepped = run_pool(prep_image, tasks, args.workers)
    receipt['timing'] = {'preprocess_seconds': time.monotonic() - t}
    idx = {p: i for i, p in enumerate(pids)}
    imgs = torch.from_numpy(np.stack([x[0] for x in prepped]))[:, None]
    tgts = torch.from_numpy(np.stack([x[1] for x in prepped]))[:, None]
    entries_by_img = {p: prepped[idx[p]][2] for p in val_ids}
    del prepped
    tr_idx = torch.tensor([idx[p] for p in tr_ids])
    va_idx = torch.tensor([idx[p] for p in val_ids])
    gt_mean_count = float(np.mean([np.mean([len(segs[r]) for r in phys_recs[p]]) for p in val_ids]))
    log(f'preprocessed {len(pids)} images in {receipt["timing"]["preprocess_seconds"]:.1f}s')
    # CUDA is touched only after the fork-based preprocessing pool has finished.
    device = torch.device(args.device if args.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
    if args.require_gpu and device.type != 'cuda':
        raise RuntimeError('GPU required but not available')
    receipt['device'] = {'type': device.type, 'cuda_devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                         if torch.cuda.is_available() else [], 'used': 'cuda:0 only' if device.type == 'cuda' else 'cpu'}
    amp = device.type == 'cuda'
    torch.backends.cudnn.benchmark = False  # E056: local 12 GB GPU, deterministic memory
    log(f'device {receipt["device"]}')

    # ---- model
    model = build_model(smp, None if args.encoder_weights == 'none' else args.encoder_weights)
    enc_bytes = b''.join(v.detach().cpu().numpy().tobytes() for v in model.encoder.state_dict().values())
    receipt['encoder_init'] = {'weights': args.encoder_weights, 'state_sha256': sha_bytes(enc_bytes),
                               'license_notes': f'smp MIT; encoder {ENCODER} (timm Apache-2.0 for tu- encoders; resnet34 torchvision BSD-3-Clause)'}
    hf = Path.home() / '.cache' / 'huggingface' / 'hub'
    if hf.exists():
        receipt['encoder_init']['hf_cache_files'] = {str(p.relative_to(hf)): sha_file(p) for p in hf.rglob('*')
                                                     if p.is_file() and p.suffix in ('.safetensors', '.bin', '.pth')}
    torch_cache = Path(torch.hub.get_dir()) / 'checkpoints'
    if torch_cache.exists():
        receipt['encoder_init']['torch_cache_files'] = {p.name: sha_file(p) for p in torch_cache.iterdir() if p.is_file()}
    model.to(device)
    if device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.amp.GradScaler('cuda', enabled=amp)
    micro, accum = (args.micro or args.batch), 1
    steps_per_epoch = math.ceil(len(tr_ids) / args.batch)
    total_steps = steps_per_epoch * args.epochs

    def lr_at(step):
        if step < args.warmup_steps:
            return (step + 1) / args.warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
    rng = random.Random(SEED)
    history = []
    yt_val = tgts[va_idx][:, 0].numpy()

    def evaluate(rec):
        tv = time.monotonic()
        probs = predict(model, imgs[va_idx], device, args.micro or args.batch, amp)
        num = den = inter = pbs = ybs = 0.0
        for j in range(len(probs)):  # per-image accumulation keeps memory small
            p, y = probs[j].astype(np.float32), yt_val[j].astype(np.float32) / 255
            num += float((p * y).sum()); den += float(p.sum() + y.sum())
            pb, yb = p > 0.5, y > 0.5  # majority = strictly more than half of the annotators
            inter += float((pb & yb).sum()); pbs += float(pb.sum()); ybs += float(yb.sum())
        rec['val_soft_dice'] = 2 * num / (den + 1e-9)
        rec['val_majority_dice'] = 2 * inter / (pbs + ybs + 1e-9)
        rec['val_seconds'] = time.monotonic() - tv
        return probs
    budget_end = args.cap_seconds - args.reserve_seconds - args.startup_seconds - args.post_seconds
    receipt.update(status='training', budget={'cap_seconds': args.cap_seconds, 'reserve_seconds': args.reserve_seconds,
                                              'startup_seconds_assumed': args.startup_seconds, 'post_seconds_assumed': args.post_seconds,
                                              'train_must_end_by_script_seconds': budget_end})
    dump(rpath, receipt)
    t_train = time.monotonic()
    epoch_est = None
    stopped_early = None
    val_probs = None
    for epoch in range(args.epochs):
        if epoch_est is not None and elapsed() + epoch_est > budget_end:
            stopped_early = f'epoch {epoch}: projected end beyond budget'
            log(stopped_early)
            break
        model.train()
        te = time.monotonic()
        perm = tr_idx[torch.randperm(len(tr_idx), generator=torch.Generator().manual_seed(SEED + epoch))]
        losses, step_times = [], []
        for s in range(0, len(perm), args.batch):
            ts = time.monotonic()
            bidx = perm[s:s + args.batch]
            opt.zero_grad(set_to_none=True)
            while True:
                try:
                    tot = 0.0
                    for m0 in range(0, len(bidx), micro):
                        mi = bidx[m0:m0 + micro]
                        x = imgs[mi].to(device, non_blocking=True).float() / 255
                        y = tgts[mi].to(device, non_blocking=True).float() / 255
                        x, y = augment(x, y, rng)
                        x = normalize(x)
                        if device.type == 'cuda':
                            x = x.contiguous(memory_format=torch.channels_last)
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                            logits = model(x)
                        loss, bce, dice = loss_fn(logits, y)
                        scaler.scale(loss * len(mi) / len(bidx)).backward()
                        tot += loss.item() * len(mi) / len(bidx)
                    break
                except torch.cuda.OutOfMemoryError:
                    if micro <= 1:
                        raise
                    opt.zero_grad(set_to_none=True)
                    torch.cuda.empty_cache()
                    micro = max(1, micro // 2)
                    receipt.setdefault('oom_fallbacks', []).append({'epoch': epoch, 'micro_batch': micro})
                    log(f'OOM: micro-batch -> {micro} (effective batch kept {args.batch} via accumulation)')
            if not math.isfinite(tot):
                raise RuntimeError(f'non-finite loss at epoch {epoch}')
            scaler.step(opt)
            scaler.update()
            sched.step()
            losses.append(tot)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            step_times.append(time.monotonic() - ts)
        rec = {'epoch': epoch, 'train_loss': float(np.mean(losses)), 'seconds': time.monotonic() - te,
               'lr': sched.get_last_lr()[0]}
        last = epoch == args.epochs - 1
        if last or (args.val_every and (epoch + 1) % args.val_every == 0) or epoch == 0:
            val_probs = evaluate(rec)
        history.append(rec)
        if epoch == 0:
            n = len(step_times)
            steady = float(np.mean(step_times[3:])) if n > 4 else float(np.mean(step_times))
            val_per_epoch = rec.get('val_seconds', 0) / max(1, args.val_every)
            epoch_est = steady * n + val_per_epoch
            projected = args.startup_seconds + elapsed() + (args.epochs - 1) * epoch_est + args.post_seconds
            receipt['first_epoch_check'] = {'epoch_seconds': rec['seconds'], 'steady_step_seconds': steady,
                                            'epoch_estimate_seconds': epoch_est, 'projected_session_seconds': projected,
                                            'limit_seconds': args.cap_seconds - args.reserve_seconds,
                                            'peak_cuda_bytes': torch.cuda.max_memory_allocated() if amp else None}
            log(f'first-epoch check {receipt["first_epoch_check"]}')
            if projected > args.cap_seconds - args.reserve_seconds:
                receipt.update(status='aborted_timing_projection', terminal=True, history=history)
                dump(rpath, receipt)
                log('ABORT: projected session time exceeds cap - reserve')
                return
        log(f'epoch {rec}')
        receipt['history'] = history
        dump(rpath, receipt)
    receipt['timing']['train_seconds'] = time.monotonic() - t_train
    receipt['stopped_early'] = stopped_early
    receipt['epochs_completed'] = len(history)
    receipt['peak_cuda_bytes'] = torch.cuda.max_memory_allocated() if amp else None

    # ---- checkpoint first, then OOF, then receipt
    ck_path = OUT / f'e004_fold{args.fold}_unet.pt'
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save({'state_dict': sd, 'config': {'arch': 'smp.Unet', 'encoder': ENCODER, 'in_channels': 1, 'classes': 1,
                                             'size': args.size, 'normalize': 'per-image mean/std on [0,1] input',
                                             'downscale': 'PIL reduce (box) 2048->size', 'smp_version': smp.__version__,
                                             'torch_version': torch.__version__, 'fold': args.fold,
                                             'epochs_completed': len(history), 'folds_csv_sha256': folds_sha}}, ck_path)
    receipt['checkpoint'] = {'path': ck_path.name, 'sha256': sha_file(ck_path), 'bytes': ck_path.stat().st_size}
    if history[-1].get('val_soft_dice') is None:  # stopped early between validation epochs
        val_probs = evaluate(history[-1])
    receipt['timing']['oof_inference_seconds'] = history[-1]['val_seconds']
    oof_path, ids_path = OUT / f'e004_fold{args.fold}_oof_probs.npy', OUT / f'e004_fold{args.fold}_oof_ids.json'
    np.save(oof_path, val_probs)
    dump(ids_path, val_ids)
    receipt['oof'] = {'path': oof_path.name, 'ids': ids_path.name, 'shape': list(val_probs.shape), 'dtype': 'float16',
                      'sha256': sha_file(oof_path)}
    receipt['metrics'] = {'val_soft_dice': history[-1].get('val_soft_dice'),
                          'val_majority_dice': history[-1].get('val_majority_dice'),
                          'final_train_loss': history[-1]['train_loss'], 'gt_mean_instances_per_image': gt_mean_count}
    receipt['status'] = 'trained_checkpoint_oof_saved'
    dump(rpath, receipt)
    log(f'checkpoint + OOF saved; dice soft {receipt["metrics"]["val_soft_dice"]} majority {receipt["metrics"]["val_majority_dice"]}')

    if args.skip_post:
        receipt['status'] = 'trained_checkpoint_oof_saved_skip_post'; receipt['terminal'] = True
        dump(rpath, receipt); log('skip-post: done'); return
    # ---- fixed-default PQ (exact local scorer semantics; parity-checked on first images)
    tp = time.monotonic()
    grid_deadline = args.cap_seconds - args.startup_seconds - 300
    default_res, info = score_settings([DEFAULT_SETTING, (0.5, 0, 64)], val_ids, val_probs, entries_by_img, device,
                                       grid_deadline, pq, parity_images=3)
    d = default_res[DEFAULT_SETTING]
    receipt['metrics']['pq_fixed_default'] = {'setting': {'threshold': 0.5, 'merge_dilation': 0, 'min_area': 0}, **d,
                                              'pred_per_image_mean': d['pred_instances'] / max(1, d['images_scored'])}
    m64 = default_res[(0.5, 0, 64)]
    receipt['metrics']['pq_default_min_area_64'] = {k: m64[k] for k in ('pq', 'tp', 'fp', 'fn', 'pred_instances')}
    receipt['metrics']['pq_default_run'] = info
    receipt['timing']['pq_default_seconds'] = time.monotonic() - tp
    dump(rpath, receipt)
    log(f'fixed-default PQ {d["pq"]:.4f} tp {d["tp"]} fp {d["fp"]} fn {d["fn"]} images {d["images_scored"]}/{len(val_ids)} parity {info["parity"]}')

    # ---- fresh-process reload + round-trip
    tr = time.monotonic()
    by_count = sorted(val_ids, key=lambda p: -np.mean([len(segs[r]) for r in phys_recs[p]]))[:3]
    expected = {}
    for p in by_count:
        up = upsample(val_probs[val_ids.index(p)], device)
        lab, n = label_instances((up > 0.5).cpu().numpy(), 0, device)
        expected[p] = n
    rt = {'checkpoint': str(ck_path), 'oof_path': str(oof_path), 'oof_ids_path': str(ids_path), 'input': str(args.input),
          'ids': by_count, 'expected_counts': expected, 'out': str(OUT), 'amp': amp}
    rt_path = OUT / 'roundtrip_request.json'
    dump(rt_path, rt)
    lib = OUT / 'lib' if EMBEDDED else Path(__file__).resolve().parent
    env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    r = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--roundtrip', str(rt_path), '--lib', str(lib),
                        '--device', args.device], capture_output=True, text=True, env=env, timeout=900)
    receipt['roundtrip'] = {'returncode': r.returncode, 'stdout_tail': r.stdout[-800:], 'stderr_tail': r.stderr[-1500:],
                            'seconds': time.monotonic() - tr}
    if (OUT / 'roundtrip.json').exists():
        rj = json.loads((OUT / 'roundtrip.json').read_text(encoding='utf-8'))
        receipt['roundtrip'].update(passed=rj.get('passed'), images=rj.get('images'), rows=rj.get('rows'),
                                    decode_mismatches=rj.get('decode_mismatches'),
                                    validator_warnings=rj.get('validator', {}).get('warnings'))
    dump(rpath, receipt)
    log(f'round-trip {receipt["roundtrip"].get("passed")} rc {r.returncode}')

    # ---- time-bounded PQ grid (tuned = selection-biased)
    tg = time.monotonic()
    settings = [(t, r_, a) for t in GRID_T for r_ in GRID_DIL for a in GRID_MIN_AREA]
    grid, ginfo = score_settings(settings, val_ids, val_probs, entries_by_img, device, grid_deadline, pq)
    rows = [{'threshold': s[0], 'merge_dilation': s[1], 'min_area': s[2],
             **{k: v[k] for k in ('pq', 'tp', 'fp', 'fn', 'tp_iou_sum', 'pred_instances', 'fragmented_gt', 'merged_pred',
                                  'images_scored')}} for s, v in grid.items()]
    rows.sort(key=lambda x: -x['pq'])
    receipt['pq_grid'] = {'complete': ginfo['complete'], 'images_scored': ginfo['images_scored'],
                          'first_image_seconds': ginfo['first_image_seconds'], 'seconds': time.monotonic() - tg,
                          'rows': rows,
                          'note': f'tuned PQ is selected on the same fold-{args.fold} OOF it is reported on: selection-biased'}
    if ginfo['complete']:
        receipt['metrics']['pq_tuned_selection_biased'] = rows[0]
    receipt['status'] = 'complete'
    receipt['terminal'] = True
    receipt['timing']['script_seconds'] = elapsed()
    dump(rpath, receipt)
    log(f'grid complete={ginfo["complete"]} best {rows[0]}')
    summary = {k: receipt.get(k) for k in ('status', 'epochs_completed', 'stopped_early', 'first_epoch_check', 'timing')}
    summary.update(pq_fixed_default=d['pq'], pq_tuned=rows[0]['pq'] if ginfo['complete'] else None,
                   dice=receipt['metrics']['val_soft_dice'], roundtrip=receipt['roundtrip'].get('passed'))
    log('SUMMARY ' + json.dumps(summary, default=str))


if __name__ == '__main__':
    main()
