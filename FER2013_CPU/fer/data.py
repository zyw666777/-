"""Preserve official splits. Never use PrivateTest to choose training samples."""
import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset

from . import CLASSES
from .common import sha256, write_json

SPLITS = ("Training", "PublicTest", "PrivateTest")
EXPECTED = {"Training": 28709, "PublicTest": 3589, "PrivateTest": 3589}


def csv_records(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if not {"emotion", "pixels", "Usage"}.issubset(reader.fieldnames or []):
            raise ValueError("CSV 必须包含 emotion,pixels,Usage。不能从混合 test 目录猜测 Public/Private 划分。")
        for row_id, row in enumerate(reader):
            values = row["pixels"].split()
            if len(values) != 2304:
                raise ValueError(f"第 {row_id+2} 行不是 48×48 像素。")
            x = np.asarray(values, dtype=np.int32)
            if (x < 0).any() or (x > 255).any():
                raise ValueError(f"第 {row_id+2} 行像素越界。")
            yield x.astype(np.uint8).reshape(48, 48), int(row["emotion"]), row["Usage"].strip(), str(row_id)


def folder_records(root):
    root = Path(root)
    for split in SPLITS:
        folder = root / split
        if not folder.is_dir():
            raise ValueError(f"缺少 {folder}；须明确提供 Training/PublicTest/PrivateTest 三个目录。")
        for name in CLASSES:
            if not (folder / name).is_dir():
                raise ValueError(f"缺少类别目录 {folder / name}")
        for file in sorted(folder.rglob("*")):
            if file.is_file() and file.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}:
                if file.parent.name not in CLASSES:
                    raise ValueError(f"未知类别目录: {file}")
                with Image.open(file) as im:
                    if im.size != (48, 48):
                        raise ValueError(f"FER 原始图像必须为 48×48: {file}")
                    x = np.array(im.convert("L"))
                yield x, CLASSES.index(file.parent.name), split, file.relative_to(root).as_posix()


def train_keep_mask(hashes, labels):
    by_hash = defaultdict(list)
    for i, h in enumerate(hashes):
        by_hash[h].append(i)
    keep = np.zeros(len(hashes), dtype=bool)
    conflicts, duplicates = 0, 0
    for indices in by_hash.values():
        if len({int(labels[i]) for i in indices}) > 1:
            conflicts += len(indices)
        else:
            keep[indices[0]] = True
            duplicates += len(indices) - 1
    return keep, conflicts, duplicates


def prepare(source, output, allow_nonstandard=False):
    source, output = Path(source), Path(output)
    if output.exists():
        raise FileExistsError(f"{output} 已存在；请使用新目录，避免覆盖数据版本。")
    groups = {s: [] for s in SPLITS}
    records = csv_records(source) if source.is_file() else folder_records(source)
    for x, y, split, row_id in records:
        if split not in groups or not 0 <= y < 7:
            raise ValueError(f"非法划分/标签: {split}/{y}")
        groups[split].append((x, y, row_id, hashlib.sha256(x.tobytes()).hexdigest()))
    counts = {s: len(v) for s, v in groups.items()}
    if any(n == 0 for n in counts.values()):
        raise ValueError("三个划分均必须非空。")
    if counts != EXPECTED and not allow_nonstandard:
        raise ValueError(f"数量与官方划分不一致: {counts}。自定义数据明确使用 --allow-nonstandard。")
    arrays = {}
    for split, rows in groups.items():
        arrays[split] = {"images": np.stack([r[0] for r in rows]),
                         "labels": np.array([r[1] for r in rows], dtype=np.int64),
                         "ids": np.array([r[2] for r in rows]), "hashes": np.array([r[3] for r in rows])}
    train = arrays["Training"]
    train["clean_mask"], conflicts, duplicates = train_keep_mask(train["hashes"], train["labels"])
    if len(set(train["labels"][train["clean_mask"]].tolist())) != 7:
        raise ValueError("清洗后的训练集须覆盖七类。")
    train_hashes = set(train["hashes"].tolist())
    public_hashes = set(arrays["PublicTest"]["hashes"].tolist())
    # No label-based filtering of validation/test. Keep full official test as primary metric.
    arrays["PublicTest"]["clean_mask"] = np.array([h not in train_hashes for h in arrays["PublicTest"]["hashes"]])
    arrays["PrivateTest"]["clean_mask"] = np.array([h not in train_hashes | public_hashes for h in arrays["PrivateTest"]["hashes"]])
    if not arrays["PublicTest"]["clean_mask"].any():
        raise ValueError("去除训练集重叠图像后验证集为空。")
    output.mkdir(parents=True)
    report = {"source": str(source.resolve()), "source_sha256": sha256(source) if source.is_file() else None,
              "official_counts_match": counts == EXPECTED, "nonstandard_dataset": counts != EXPECTED,
              "class_order": CLASSES, "train_conflicting_samples_removed": conflicts,
              "train_duplicate_samples_removed": duplicates, "splits": {},
              "protocol": "Training-only deduplication; PublicTest selects models; full PrivateTest final evaluation. Clean test is supplemental and excludes pixel overlap with Training or PublicTest."}
    for split, a in arrays.items():
        file = output / f"{split}.npz"
        np.savez_compressed(file, **a)
        report["splits"][split] = {"total": len(a["labels"]), "clean": int(a["clean_mask"].sum()),
                                   "class_counts": dict(Counter(CLASSES[int(y)] for y in a["labels"])),
                                   "sha256": sha256(file)}
    write_json(output / "audit.json", report)
    print(f"Prepared {counts}; audit: {output / 'audit.json'}", flush=True)
    return report


class FerDataset(Dataset):
    def __init__(self, data, split, transform, clean=True):
        if split not in SPLITS:
            raise ValueError(split)
        with np.load(Path(data) / f"{split}.npz", allow_pickle=False) as a:
            mask = a["clean_mask"] if clean else np.ones(len(a["labels"]), dtype=bool)
            self.images = a["images"][mask]
            self.labels = a["labels"][mask]
            self.ids = a["ids"][mask]
            self.clean_mask = a["clean_mask"][mask]
        if not len(self.labels):
            raise ValueError(f"空数据集: {split}")
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.fromarray(self.images[index]).convert("RGB")
        return self.transform(image), int(self.labels[index])


def main():
    p = argparse.ArgumentParser(description="导入官方 FER2013 CSV 或三划分图片目录。")
    p.add_argument("--source", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--allow-nonstandard", action="store_true")
    a = p.parse_args()
    prepare(a.source, a.out, a.allow_nonstandard)


if __name__ == "__main__":
    main()
