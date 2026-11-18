import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch

from . import CLASSES


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_checkpoint(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temp)
    temp.replace(path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def device_from(name="auto"):
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用。请安装匹配显卡的 PyTorch，或仅用 --device cpu 做功能检查。")
    return device


def metrics(labels, probabilities):
    labels = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if not len(labels) or p.shape != (len(labels), 7):
        raise ValueError("需要非空标签和 [N,7] 概率。")
    if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1, atol=1e-4):
        raise ValueError("概率必须有限、非负且每行和为 1。")
    pred = p.argmax(1)
    cm = np.bincount(labels * 7 + pred, minlength=49).reshape(7, 7)
    tp = cm.diagonal()
    support = cm.sum(1)
    recall = np.divide(tp, support, out=np.zeros(7, dtype=float), where=support != 0)
    precision = np.divide(tp, cm.sum(0), out=np.zeros(7, dtype=float), where=cm.sum(0) != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(7), where=precision + recall != 0)
    n = len(labels)
    acc = float((pred == labels).mean())
    z = 1.959963984540054
    center = (acc + z*z/(2*n)) / (1 + z*z/n)
    half = z * np.sqrt(acc*(1-acc)/n + z*z/(4*n*n)) / (1 + z*z/n)
    return {"samples": n, "correct": int((pred == labels).sum()), "accuracy": acc,
            "accuracy_wilson_95": [float(center-half), float(center+half)],
            "macro_f1": float(f1.mean()), "balanced_accuracy": float(recall.mean()),
            "nll": float(-np.log(np.clip(p[np.arange(n), labels], 1e-12, 1)).mean()),
            "class_order": CLASSES, "confusion_matrix": cm.tolist(),
            "per_class": {c: {"precision": float(precision[i]), "recall": float(recall[i]),
                               "f1": float(f1[i]), "support": int(support[i])} for i, c in enumerate(CLASSES)}}


def environment_info():
    import platform
    import timm
    import torchvision
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "torchvision": str(torchvision.__version__), "timm": timm.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cpu_count": os.cpu_count()}
