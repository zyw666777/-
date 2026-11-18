"""Download an immutable public FER2013 CSV mirror and verify file hashes."""
import argparse
import shutil
import tarfile
import urllib.request
from pathlib import Path

from .common import sha256, write_json

REVISION = "5b1d92d29dea477a4c13af5f928b2baee3744951"
URL = f"https://huggingface.co/datasets/DerrickUnleashed/FER-2013/resolve/{REVISION}/fer2013.tar.gz"
ARCHIVE_SHA = "bdc255c109a1bbf994e6a7dd02a6f4566eb21beaece7643cd988166ee3527a44"
CSV_SHA = "3b8d9617d1017f34733c8f2474d7784c563ce86c40a86ac12c2d37cc968f871b"


def fetch(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    archive, csv = out / "fer2013.tar.gz", out / "fer2013.csv"
    if not archive.exists():
        partial = archive.with_suffix(".gz.part")
        with urllib.request.urlopen(URL, timeout=60) as response, partial.open("wb") as target:
            shutil.copyfileobj(response, target, length=1024 * 1024)
        if sha256(partial) != ARCHIVE_SHA:
            raise ValueError("数据压缩包校验失败，未导入。")
        partial.replace(archive)
    if sha256(archive) != ARCHIVE_SHA:
        raise ValueError("已有压缩包校验不符，请使用新的输出目录。")
    if not csv.exists():
        with tarfile.open(archive, "r:gz") as bundle:
            member = bundle.getmember("fer2013/fer2013.csv")
            if not member.isfile() or member.size != 301072766:
                raise ValueError("CSV 成员类型或大小不符。")
            partial = csv.with_suffix(".csv.part")
            with bundle.extractfile(member) as source, partial.open("wb") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
        if sha256(partial) != CSV_SHA:
            raise ValueError("CSV 校验失败，未导入。")
        partial.replace(csv)
    if sha256(csv) != CSV_SHA:
        raise ValueError("已有 CSV 校验不符，请使用新的输出目录。")
    write_json(out / "public_data_source.json", {
        "source": URL, "revision": REVISION, "archive_sha256": ARCHIVE_SHA,
        "csv_sha256": CSV_SHA, "provenance": "Community mirror; original FER2013 labels and Usage retained.",
    })
    print(csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/raw")
    fetch(parser.parse_args().out)
