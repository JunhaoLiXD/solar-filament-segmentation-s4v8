"""Validate variable-row filament instance CSV against the actual test manifest."""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath

from gpu_smoke_audit import parse_compressed_counts

# Valid high-frequency 2048x2048 masks can exceed Python CSV's 128 KiB default.
CSV_FIELD_LIMIT = 64 * 1024 * 1024


def test_ids_from_manifest(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise ValueError("A complete official file manifest is required")
    ids = []
    for row in manifest["files"]:
        file = PurePosixPath(row["name"])
        if "test_images" in file.parts and file.suffix.lower() in {".jpg", ".jpeg"}:
            ids.append(file.stem)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Manifest test IDs are empty or ambiguous")
    return set(ids)


def validate_rle(value):
    from pycocotools import mask as coco_mask
    if not value or value != value.strip() or value.startswith(("b'", 'b"', "{")):
        raise ValueError("RLE must be the compressed counts string alone")
    counts = parse_compressed_counts(value)
    if not counts or sum(counts) != 2048 * 2048:
        raise ValueError("RLE runs must cover exactly 2048 x 2048 pixels")
    # The bounded parser above rejects malformed sizes before entering native code.
    decoded = coco_mask.decode({"size": [2048, 2048], "counts": value.encode("ascii")})
    if decoded.shape != (2048, 2048):
        raise ValueError("RLE decoded dimensions do not match [2048, 2048]")
    # Canonical roundtrip prevents accepting trailing garbage / alternate byte syntax.
    canonical = coco_mask.encode(decoded)["counts"].decode("ascii")
    if canonical != value:
        raise ValueError("RLE is not canonical compressed COCO counts")
    return int(decoded.sum())


def validate_submission(path, manifest=None, *, allowed_ids=None, reject_empty_masks=False, reject_all_empty=False):
    csv.field_size_limit(max(csv.field_size_limit(), CSV_FIELD_LIMIT))
    if allowed_ids is None:
        if manifest is None:
            manifest = Path(__file__).resolve().parents[1] / "research/data-file-manifest.json"
        test_ids = test_ids_from_manifest(manifest)
    else:
        values = list(allowed_ids)
        if not values or len(values) != len(set(values)) or any(not isinstance(v, str) or not v or v != v.strip() or '_' in v or any(c.isspace() for c in v) for v in values):
            raise ValueError("Explicit allowed held-out IDs must be nonempty, unique physical-image stems")
        test_ids = set(values)
    seen, predicted, empty_masks, total_area = set(), set(), 0, 0
    with Path(path).open("r", newline="", encoding="utf-8-sig") as stream:
        reader = csv.DictReader(stream, strict=True)
        if reader.fieldnames != ["filament_id", "segmentation_rle"]:
            raise ValueError("Expected exactly filament_id,segmentation_rle in that order")
        for line, row in enumerate(reader, 2):
            try:
                if None in row or None in row.values():
                    raise ValueError("Wrong field count")
                instance = row["filament_id"]
                if instance in seen:
                    raise ValueError(f"Duplicate instance ID: {instance}")
                if instance != instance.strip() or "_" not in instance:
                    raise ValueError("Instance ID needs original image stem and nonempty underscore suffix")
                image, suffix = instance.split("_", 1)
                if image not in test_ids or not suffix or any(c.isspace() for c in suffix):
                    raise ValueError(f"Foreign image ID or invalid instance suffix: {instance}")
                area = validate_rle(row["segmentation_rle"])
                if reject_empty_masks and area == 0:
                    raise ValueError("Candidate policy rejects empty instance masks")
                empty_masks += area == 0
                total_area += area
                seen.add(instance)
                predicted.add(image)
            except (ValueError, TypeError, KeyError, UnicodeError, OverflowError) as exc:
                raise ValueError(f"CSV line {line}: {exc}") from exc
    if reject_all_empty and total_area == 0:
        raise ValueError("Candidate policy rejects all-empty output (including no rows)")
    missing = sorted(test_ids - predicted)
    warnings = []
    if missing:
        warnings.append(f"{len(missing)} test images have no predicted rows; server coverage/empty-image conventions remain unverified")
    if empty_masks:
        warnings.append(f"{empty_masks} empty instance masks; server acceptance remains unverified")
    return {"valid": True, "rows": len(seen), "predicted_images": len(predicted),
            "test_images": len(test_ids), "missing_image_ids": missing,
            "empty_masks": empty_masks, "warnings": warnings,
            "id_scope": "explicit held-out IDs" if allowed_ids is not None else "official test manifest",
            "candidate_policy": {"reject_empty_masks": reject_empty_masks, "reject_all_empty": reject_all_empty},
            "scope": "format validation only; no exact row-count, full coverage or server-score guarantee"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--manifest", type=Path, default=Path(__file__).resolve().parents[1] / "research/data-file-manifest.json")
    parser.add_argument("--allowed-heldout-ids", type=Path, help="JSON list of permitted physical image stems; replaces test manifest ID scope")
    parser.add_argument("--reject-empty-masks", action="store_true")
    parser.add_argument("--reject-all-empty", action="store_true")
    args = parser.parse_args()
    try:
        ids = json.loads(args.allowed_heldout_ids.read_text(encoding="utf-8")) if args.allowed_heldout_ids else None
        if ids is not None and not isinstance(ids, list):
            raise ValueError("Allowed held-out IDs file must contain a JSON list")
        print(json.dumps(validate_submission(args.csv, args.manifest, allowed_ids=ids,
                                             reject_empty_masks=args.reject_empty_masks, reject_all_empty=args.reject_all_empty), indent=2))
    except (ValueError, csv.Error, OSError) as exc:
        parser.exit(1, f"Validation failed: {exc}\n")
