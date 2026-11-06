"""Original offline Kaggle audit + tiny training smoke test; never submits or infers test."""
import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import traceback

START = time.monotonic()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def terminal_receipt(out, status, details):
    path = out / "audit.json"
    try:
        audit = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        audit = {}
    audit.update(status=status, baseline_ready=False, full_audit_complete=False,
                 total_seconds=time.monotonic() - START, terminal_details=details)
    dump(path, audit)


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_name(path):
    match = re.fullmatch(r"(\d{8})(\d{6})([A-Za-z]{2})", path.stem)
    if not match:
        raise ValueError(f"Unexpected physical image name: {path.name}")
    from datetime import datetime
    datetime.strptime(match[1] + match[2], "%Y%m%d%H%M%S")
    return match[1], match[3]


def main(args):
    # Import heavy dependencies only after the hard wall-clock watchdog is active.
    import numpy as np
    import PIL
    from PIL import Image
    import torch
    from torch import nn
    from pycocotools import mask as coco_mask

    out = args.output
    effective_config = {"steps": args.steps, "wall_seconds": args.timeout,
                        "input": str(args.input), "output": str(args.output),
                        "requested_accelerator": "NvidiaTeslaT4", "server_timeout_expected_seconds": 450,
                        "maximum_allocated_devices": 2, "maximum_aggregate_gpu_hours": 0.25,
                        "architecture": "original Conv1-8-8-1", "seed": 2026,
                        "image_resize": [256, 256], "maximum_batch": 4}
    audit = {"purpose": "environment and partial data audit only; no baseline CV", "status": "running",
             "baseline_ready": False, "full_audit_complete": False,
             "stages": {"structural_audit": "running", "mask_decoding": "pending", "training": "pending", "checkpoint": "pending"},
             "outstanding_checks": ["full mask decoding and foreground distributions", "category/bbox/area/license integrity", "near-duplicate audit", "final balanced folds", "official metric parity", "held-out inference and checkpoint reload/export"],
             "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "effective_config": effective_config,
             "config_sha256": hashlib.sha256(json.dumps(effective_config, sort_keys=True).encode()).hexdigest(),
             "limits": {"steps": args.steps, "wall_seconds": args.timeout},
             "versions": {"torch": torch.__version__, "numpy": np.__version__, "Pillow": PIL.__version__, "pycocotools": importlib.metadata.version("pycocotools")}}
    dump(out / "audit.json", audit)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: refusing CPU/local training fallback")
    audit["cuda"] = {"version": torch.version.cuda, "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}
    dump(out / "audit.json", audit)
    if torch.cuda.device_count() not in (1, 2):
        raise RuntimeError("This T4 diagnostic requires one or two allocated CUDA devices")
    candidates = []
    for path in args.input.rglob("*.json"):
        if "train" in path.name.lower() and "annotation" in path.name.lower():
            candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one train annotation JSON; found {candidates}")
    annotation_path = candidates[0]
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("images"), list) or not isinstance(data.get("annotations"), list):
        raise ValueError("Annotation JSON must contain COCO images and annotations lists")
    if not data["images"] or not data["annotations"]:
        raise ValueError("Empty training images or annotations")
    files = {}
    split_for = {}
    for path in args.input.rglob("*"):
        if path.suffix.lower() not in {".jpg", ".jpeg"}:
            continue
        dirs = {p.lower() for p in path.parts}
        split = "train" if "train_images" in dirs else "test" if "test_images" in dirs else None
        if split is None:
            continue
        if path.name in files or path.stem in split_for:
            raise ValueError(f"Ambiguous duplicate mounted image name: {path}")
        files[path.name] = path
        split_for[path.stem] = split
    if set(split_for.values()) != {"train", "test"}:
        raise ValueError("Expected mounted train_images and test_images JPEG directories")
    physical = {}
    for name, path in sorted(files.items()):
        day, station = parse_name(path)
        with Image.open(path) as im:
            if im.size != (2048, 2048):
                raise ValueError(f"Invalid image dimensions: {path}: {im.size}")
            if im.mode not in ("L", "RGB"):
                raise ValueError(f"Unexpected JPEG mode {im.mode}: {path}")
            if im.mode == "RGB":
                rgb = np.asarray(im)
                if not (np.array_equal(rgb[:, :, 0], rgb[:, :, 1]) and np.array_equal(rgb[:, :, 1], rgb[:, :, 2])):
                    raise ValueError(f"Non-grayscale RGB image: {path}")
            pixels = np.asarray(im.convert("L"))
            digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        physical[path.stem] = {"physical_id": path.stem, "split": split_for[path.stem], "day": day,
                               "station": station, "pixel_sha256": digest, "path": str(path)}
    records = {}
    counts = {}
    annotations = {}
    for row in data["images"]:
        key = str(row["id"])
        name = Path(row["file_name"]).name
        if key in records or name not in files or split_for.get(Path(name).stem) != "train":
            raise ValueError(f"Duplicate image record ID or unresolved train image: {key}, {name}")
        if (row.get("width"), row.get("height")) != (2048, 2048):
            raise ValueError(f"Bad COCO dimensions: {key}")
        records[key] = row
        counts[key] = 0
        annotations[key] = []
    seen_annotations = set()
    segmentation_types = {}
    for row in data["annotations"]:
        key, aid = str(row["image_id"]), str(row["id"])
        if key not in records or aid in seen_annotations:
            raise ValueError(f"Orphan or duplicate annotation: {aid}, image {key}")
        seen_annotations.add(aid)
        seg = row.get("segmentation")
        if isinstance(seg, list) and seg:
            for polygon in seg:
                if not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2:
                    raise ValueError(f"Invalid polygon structure: {aid}")
                if not all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 2048 for v in polygon):
                    raise ValueError(f"Nonfinite/out-of-bounds polygon: {aid}")
            kind = "polygon"
        elif isinstance(seg, dict) and seg.get("size") == [2048, 2048] and isinstance(seg.get("counts"), (list, str)):
            kind = "rle"
            # Validate every native RLE, rather than passing malformed buffers to C.
            counts_value = seg["counts"]
            if isinstance(counts_value, str):
                counts_value = parse_compressed_counts(counts_value)
            if not counts_value or any(type(v) is not int or v < 0 for v in counts_value) or sum(counts_value) != 2048 * 2048:
                raise ValueError(f"Invalid RLE counts: {aid}")
        else:
            raise ValueError(f"Unsupported or empty segmentation: {aid}")
        segmentation_types[kind] = segmentation_types.get(kind, 0) + 1
        annotations[key].append(row)
        counts[key] += 1
    unreferenced = sorted(set(p for p, v in physical.items() if v["split"] == "train") - {Path(r["file_name"]).stem for r in records.values()})
    if unreferenced:
        raise ValueError(f"Train files absent from annotations: {unreferenced[:10]}")
    # Candidate connected components join same-day observations and exact pixel duplicates.
    parent = {p: p for p in physical}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    days, hashes = {}, {}
    for p, row in physical.items():
        for value, groups in ((row["day"], days), (row["pixel_sha256"], hashes)):
            if value in groups:
                union(p, groups[value][0])
            groups.setdefault(value, []).append(p)
    fold_rows, manifest = [], []
    for key, row in sorted(records.items()):
        p = Path(row["file_name"]).stem
        group = find(p)
        fold = int(hashlib.sha256(group.encode()).hexdigest()[:8], 16) % 5
        fold_rows.append({"image_id": key, "physical_id": p, "group": group, "fold": fold})
        manifest.append({"image_id": key, **physical[p], "annotation_count": counts[key], "license": row.get("license")})
    write_csv(out / "manifest.csv", manifest)
    write_csv(out / "folds_candidate.csv", fold_rows)
    dump(out / "group_report.json", {"policy": "candidate only: connected same-day / exact pixel duplicate groups; not final CV",
         "exact_duplicate_groups": [v for v in hashes.values() if len(v) > 1],
         "multi_image_day_groups": {k: v for k, v in days.items() if len(v) > 1},
         "physical_images": list(physical.values()),
         "train_test_day_overlap": [k for k, v in days.items() if len({physical[p]["split"] for p in v}) > 1]})
    audit.update({"annotation_file": str(annotation_path), "annotation_sha256": hashlib.sha256(annotation_path.read_bytes()).hexdigest(),
                  "physical_train": sum(v["split"] == "train" for v in physical.values()),
                  "physical_test": sum(v["split"] == "test" for v in physical.values()),
                  "annotator_image_records": len(records), "annotation_count": len(seen_annotations),
                  "segmentation_types": segmentation_types, "licenses": data.get("licenses"),
                  "fold_record_counts": {i: sum(r["fold"] == i for r in fold_rows) for i in range(5)},
                  "audit_seconds": time.monotonic() - START})
    dump(out / "audit.json", audit)
    # One annotator's union mask per chosen physical image; this is only a plumbing test.
    xs, ys, selected, chosen = [], [], [], set()
    decoded_annotation_records = 0
    for key, row in sorted(records.items()):
        p = Path(row["file_name"]).stem
        if p in chosen or counts[key] == 0:
            continue
        chosen.add(p)
        mask = np.zeros((2048, 2048), dtype=np.uint8)
        for ann in annotations[key]:
            seg = ann["segmentation"]
            if isinstance(seg, list):
                rle = coco_mask.merge(coco_mask.frPyObjects(seg, 2048, 2048))
            elif isinstance(seg["counts"], list):
                rle = coco_mask.frPyObjects(seg, 2048, 2048)
            else:
                rle = {"size": [2048, 2048], "counts": seg["counts"].encode("ascii")}
            decoded = coco_mask.decode(rle)
            if decoded.shape != mask.shape:
                raise ValueError(f"Decoded shape mismatch: {ann['id']}")
            mask |= decoded
            decoded_annotation_records += 1
        with Image.open(files[Path(row["file_name"]).name]) as im:
            xs.append(np.asarray(im.convert("L").resize((256, 256), Image.Resampling.BILINEAR), dtype=np.float32) / 255)
        ys.append(np.asarray(Image.fromarray(mask).resize((256, 256), Image.Resampling.NEAREST), dtype=np.float32))
        selected.append(key)
        if len(xs) == 4:
            break
    if len(xs) < 2 or not any(np.any(y) for y in ys):
        raise ValueError("Need at least two real annotated images and nonempty targets")
    audit.update(decoded_annotation_records=decoded_annotation_records,
                 decoded_annotator_image_records=len(selected))
    audit["stages"].update(structural_audit="partial", mask_decoding="selected_records_only", training="running")
    dump(out / "audit.json", audit)
    torch.manual_seed(2026)
    torch.cuda.manual_seed_all(2026)
    model = nn.Sequential(nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 8, 3, padding=1),
                          nn.ReLU(), nn.Conv2d(8, 1, 1)).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.from_numpy(np.stack(xs)[:, None]).cuda()
    y = torch.from_numpy(np.stack(ys)[:, None]).cuda()
    training_start = time.monotonic()
    losses, step_seconds = [], []
    for step in range(args.steps):
        if time.monotonic() - START > args.timeout - 30:
            raise TimeoutError("Stopped before hard timeout; insufficient training time")
        tick = time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.binary_cross_entropy_with_logits(model(x), y)
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite training loss")
        loss.backward()
        if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
            raise ValueError("Nonfinite training gradients")
        optimizer.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach().cpu()))
        step_seconds.append(time.monotonic() - tick)
    torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "architecture": "original Conv1-8-8-1", "smoke_only": True, "training_image_ids": selected}, out / "tiny_checkpoint.pt")
    audit["stages"].update(training="success", checkpoint="saved_not_reload_validated")
    audit.update({"status": "environment_smoke_success", "training_ids": selected, "training_steps": len(losses), "losses": losses,
                  "step_seconds": step_seconds, "train_seconds": time.monotonic() - training_start,
                  "total_seconds": time.monotonic() - START, "peak_cuda_memory_bytes": torch.cuda.max_memory_allocated(),
                  "test_inference_seconds": 0, "cv_score": None})
    dump(out / "audit.json", audit)
    print(json.dumps(audit), flush=True)


def parse_compressed_counts(value):
    """Bounded Python COCO counts parser, before any native decoder."""
    counts, pos = [], 0
    while pos < len(value):
        number, shift = 0, 0
        while True:
            if pos >= len(value) or shift > 30:
                raise ValueError("Truncated or oversized compressed RLE")
            char = ord(value[pos]) - 48
            pos += 1
            if not 0 <= char <= 63:
                raise ValueError("Invalid compressed RLE character")
            number |= (char & 31) << shift
            shift += 5
            if not char & 32:
                if char & 16:
                    number |= -1 << shift
                break
        if len(counts) > 2:
            number += counts[-2]
        if number < 0 or number > 2048 * 2048:
            raise ValueError("RLE run out of bounds")
        counts.append(number)
        if len(counts) > 2048 * 2048 + 1:
            raise ValueError("Too many RLE runs")
    return counts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working"))
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=360)
    args = parser.parse_args()
    if not 1 <= args.steps <= 20 or not 60 <= args.timeout <= 360:
        parser.error("steps must be 1..20; timeout 60..360 (server limit must be 450)")
    if not Path("/kaggle").is_dir() or args.input.resolve() != Path("/kaggle/input").resolve():
        parser.error("This smoke test is Kaggle-only and must use /kaggle/input")
    args.output.mkdir(parents=True, exist_ok=True)
    def hard_stop():
        details = {"status": "hard_timeout", "elapsed_seconds": time.monotonic() - START}
        terminal_receipt(args.output, "hard_timeout", details)
        dump(args.output / "timeout.json", details)
        os._exit(124)
    timer = threading.Timer(max(1, args.timeout - (time.monotonic() - START)), hard_stop)
    timer.daemon = True
    timer.start()
    try:
        main(args)
    except Exception:
        details = {"status": "failed", "elapsed_seconds": time.monotonic() - START,
                   "traceback": traceback.format_exc()}
        terminal_receipt(args.output, "failed", details)
        dump(args.output / "failure.json", details)
        raise
    finally:
        timer.cancel()
