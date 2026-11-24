from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from PIL import Image

SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO = re.compile(r"^(Vid_(?:Ca|Pr)_(?:evening|night))(?=$|[^A-Za-z])", re.I)
ORIGIN_SUFFIX = re.compile(r"_(?:jpe?g|png)\.rf\.|\.rf\.", re.I)


def write_csv(path, fields, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def file_digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def locate_members(archive):
    indexed = {}
    for member in archive.infolist():
        if member.is_dir():
            continue
        parts = PurePosixPath(member.filename).parts
        if len(parts) < 3 or parts[-3] not in ("images", "labels") or parts[-2] not in SPLITS:
            continue
        key = "/".join(parts[-3:])
        if key in indexed:
            raise ValueError(f"Duplicate member for {key}")
        indexed[key] = member
    return indexed


def parse_label(payload, path, issues):
    lines = [line.strip() for line in payload.decode("utf-8-sig").splitlines() if line.strip()]
    seen = set()
    for line_number, line in enumerate(lines, 1):
        try:
            box = tuple(map(float, line.split()))
            cls, x, y, w, h = box
            if not all(map(math.isfinite, box)):
                raise ValueError("non-finite coordinate")
            if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError("unexpected class or invalid normalized box")
        except ValueError as exc:
            issues.append({"path": path, "line": line_number, "kind": str(exc)})
            continue
        if box in seen:
            issues.append({"path": path, "line": line_number, "kind": "duplicate annotation row"})
        if max(w / 2 - x, h / 2 - y, x + w / 2 - 1, y + h / 2 - 1) > 1e-6:
            issues.append({"path": path, "line": line_number, "kind": "annotation extends beyond image border"})
        seen.add(box)
    return len(lines)


def build_manifest(path, historical):
    rows, issues = [], []
    by_split = {split: Counter() for split in SPLITS}
    historic_mismatches = 0
    with ZipFile(path) as archive:
        members = locate_members(archive)
        used_labels = set()
        for rel, member in sorted(members.items()):
            p = PurePosixPath(rel)
            if p.parts[0] != "images" or p.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            split = p.parts[1]
            label_rel = f"labels/{split}/{p.stem}.txt"
            label_bytes = archive.read(members[label_rel]) if label_rel in members else None
            if label_bytes is None:
                issues.append({"path": rel, "line": "", "kind": "missing label"})
            else:
                used_labels.add(label_rel)
            box_count = parse_label(label_bytes, label_rel, issues) if label_bytes is not None else 0
            image_bytes = archive.read(member)
            with Image.open(io.BytesIO(image_bytes)) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                pixel_hash = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            origin = ORIGIN_SUFFIX.split(p.name, 1)[0] if ORIGIN_SUFFIX.search(p.name) else p.stem
            video = VIDEO.match(origin)
            candidate = historical.get(rel, {})
            historical_group = ""
            if candidate:
                if candidate["pixel_sha256"] == pixel_hash:
                    historical_group = candidate["group_id"]
                else:
                    historic_mismatches += 1
            rows.append({
                "image_id": p.stem,
                "split": split,
                "image_path": rel,
                "label_path": label_rel if label_bytes is not None else "",
                "bbox_count": box_count,
                "width": width,
                "height": height,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "decoded_rgb_sha256": pixel_hash,
                "label_sha256": hashlib.sha256(label_bytes).hexdigest() if label_bytes is not None else "",
                "source_dataset": "unverified",
                "filename_origin_candidate": origin,
                "video_prefix_candidate": video.group(1) if video else "",
                "verified_capture_group_id": "",
                "historical_candidate_group_id": historical_group,
            })
            by_split[split].update(images=1, boxes=box_count)
        for rel in sorted(members):
            if rel.startswith("labels/") and rel.endswith(".txt") and rel not in used_labels:
                issues.append({"path": rel, "line": "", "kind": "orphan label"})
    return rows, by_split, issues, historic_mismatches


def cross_split_groups(rows, columns):
    groups, pairs = [], []
    for method in columns:
        index = defaultdict(list)
        for row in rows:
            if row[method]:
                index[row[method]].append(row)
        for group_id, items in index.items():
            counts = Counter(row["split"] for row in items)
            if len(counts) < 2:
                continue
            groups.append({
                "method": method,
                "group_id": group_id,
                "train": counts["train"],
                "val": counts["val"],
                "test": counts["test"],
                "total": len(items),
            })
            pairs.extend({
                "method": method, "group_id": group_id,
                "split": row["split"], "image_path": row["image_path"],
            } for row in items)
    return groups, pairs


def run():
    p = argparse.ArgumentParser(description="Audit an archived YOLO image split without modifying its source.")
    p.add_argument("dataset", type=Path, help="NPD_YOLO.zip with images and labels grouped by split")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--historical-groups", type=Path, help="Optional older candidate groups with decoded-pixel hashes")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    historical = {}
    if args.historical_groups:
        historical = {
            row["path"]: row for row in json.loads(args.historical_groups.read_text(encoding="utf-8-sig"))
        }
    rows, counts, issues, historic_mismatches = build_manifest(args.dataset, historical)
    assert rows, "No dataset images were found"
    methods = (
        "image_sha256", "decoded_rgb_sha256", "filename_origin_candidate",
        "video_prefix_candidate", "historical_candidate_group_id",
    )
    groups, members = cross_split_groups(rows, methods)
    write_csv(args.out / "split_manifest.csv", list(rows[0]), rows)
    write_csv(args.out / "cross_split_groups.csv", ["method", "group_id", *SPLITS, "total"], groups)
    write_csv(args.out / "cross_split_group_members.csv", ["method", "group_id", "split", "image_path"], members)
    write_csv(args.out / "label_review_flags.csv", ["path", "line", "kind"], issues)
    report = {
        "dataset_archive": args.dataset.name,
        "dataset_archive_sha256": file_digest(args.dataset),
        "splits": {s: dict(c) for s, c in counts.items()},
        "totals": {"images": len(rows), "boxes": sum(r["bbox_count"] for r in rows)},
        "label_review_flags": dict(Counter(r["kind"] for r in issues)),
        "filename_video_prefix_coverage_images": sum(bool(r["video_prefix_candidate"]) for r in rows),
        "historical_candidate_group_coverage_images": sum(bool(r["historical_candidate_group_id"]) for r in rows),
        "historical_pixel_hash_mismatches": historic_mismatches,
        "cross_split_groups": {method: sum(r["method"] == method for r in groups) for method in methods},
        "cross_split_images": {
            method: len({r["image_path"] for r in members if r["method"] == method}) for method in methods
        },
        "verified_source_dataset_coverage_images": 0,
        "verified_capture_group_coverage_images": 0,
        "limitations": [
            "Filename stems and video prefixes are inferred from names.",
            "Historical groups are heuristic candidates on an older archive and are transferred only where relative path and decoded RGB hash both match.",
            "A candidate group is not a confirmed physical road segment or verified capture sequence.",
            "The audit does not establish independence at the video/road-segment level or verify source attribution per image.",
        ],
    }
    (args.out / "archive_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()

