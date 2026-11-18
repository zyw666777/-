"""Recalculate reported accuracy against the original FER2013 labels."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True, help='Prepared FER2013 data directory')
    args = parser.parse_args()
    report = json.loads((ROOT/'metrics.json').read_text())
    data_file = args.data/f"{report['split']}.npz"
    with data_file.open('rb') as source:
        digest = hashlib.file_digest(source,'sha256').hexdigest()
    if digest != report['test_data_sha256']:
        # ZIP timestamps and NumPy versions can change binary NPZ composition.
        # Label/ID/clean-mask hashes below verify the semantic contents.
        print('数据压缩文件字节不同，继续核对样本 ID、标签与去重标记。')
    with np.load(data_file,allow_pickle=False) as data,np.load(ROOT/'evaluation_predictions.npz',allow_pickle=False) as pred:
        if not np.array_equal(pred['ids'],data['ids']):
            raise ValueError('评测样本 ID 或顺序不一致')
        for name in ('labels','clean_mask','hashes'):
            array = np.ascontiguousarray(data[name])
            if array.dtype.str != report['data_arrays'][name]['dtype']:
                raise ValueError(f'评测数据类型不一致 {name}')
            if hashlib.sha256(array.tobytes()).hexdigest() != report['data_arrays'][name]['sha256']:
                raise ValueError(f'评测数据不一致 {name}')
        correct = pred['probabilities'].argmax(1) == data['labels']
        clean = data['clean_mask']
        if not (len(correct) == report['samples'] and int(correct.sum()) == report['correct']):
            raise ValueError('复核分数与报告不一致')
        print(f"{report['split']} {int(correct.sum())} / {len(correct)} = {correct.mean():.4%}")
        print(f"去重部分 {int(correct[clean].sum())} / {int(clean.sum())} = {correct[clean].mean():.4%}")


if __name__ == '__main__':
    main()
