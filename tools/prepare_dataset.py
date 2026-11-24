from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from audit_dataset import locate_members

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description="Reconstruct selected splits from a matching NPD_YOLO.zip.")
    p.add_argument("archive", type=Path)
    p.add_argument("--splits", nargs="+", choices=("train", "val", "test"), default=["train", "val", "test"])
    args = p.parse_args()
    with (ROOT / "data_audit" / "split_manifest.csv").open(newline="", encoding="utf-8-sig") as f:
        rows = [row for row in csv.DictReader(f) if row["split"] in args.splits]
    expected = {}
    for row in rows:
        expected[row["image_path"]] = row["image_sha256"]
        if row["label_path"]:
            expected[row["label_path"]] = row["label_sha256"]
    with ZipFile(args.archive) as z:
        members = locate_members(z)
        for rel, digest in expected.items():
            parts = PurePosixPath(rel).parts
            if len(parts) != 3 or parts[-1] in ("", ".", "..") or any(c in parts[-1] for c in "\\:"):
                raise ValueError(f"Unsafe manifest path: {rel}")
            data = z.read(members[rel])
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError(f"Archive file does not match reference manifest: {rel}")
        for rel in expected:
            target = ROOT / "data" / rel
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected[rel]:
                raise FileExistsError(f"Existing file differs: {target}")
        for rel in expected:
            target = ROOT / "data" / rel
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(members[rel]))
    for split in args.splits:
        for kind in ("images", "labels"):
            folder = ROOT / "data" / kind / split
            names = {PurePosixPath(rel).name for rel in expected if rel.startswith(f"{kind}/{split}/")}
            if any(p.is_file() and p.suffix.lower() in {".jpg",".jpeg",".png",".bmp",".webp",".txt"}
                   and p.name not in names for p in folder.iterdir()):
                raise ValueError(f"Unexpected data in {folder}")
    (ROOT / "test_only.yaml").write_text(json.dumps({
        "path": str(ROOT / "data"),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {"0": "pothole"},
    }, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {len(rows)} image/label pairs for {', '.join(args.splits)}")


if __name__ == "__main__":
    main()

