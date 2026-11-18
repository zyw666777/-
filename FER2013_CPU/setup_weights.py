"""Fetch official pretrained model files and verify their recorded SHA-256."""
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from urllib.request import urlopen
import zipfile

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT/'weights'

def sha256(file):
    with file.open('rb') as source:
        return hashlib.file_digest(source,'sha256').hexdigest()


def download(url, filename):
    with urlopen(url,timeout=90) as response, filename.open('wb') as target:
        shutil.copyfileobj(response,target,length=1024*1024)


def main():
    WEIGHTS.mkdir(exist_ok=True)
    sources = json.loads((ROOT/'weight_sources.json').read_text())
    for model in sources:
        destination = WEIGHTS/model['filename']
        expected = model['sha256']
        if destination.exists() and sha256(destination) == expected:
            print(f'校验通过 {destination.name}',flush=True)
            continue
        if 'url' not in model:
            raise FileNotFoundError(f'缺少包内权重 {destination.name}，请重新解压。')
        print(f'下载 {destination.name}',flush=True)
        with tempfile.TemporaryDirectory(dir=WEIGHTS) as folder:
            temporary = Path(folder)
            source = temporary/'download'
            download(model['url'],source)
            if 'zip_member' in model:
                target = temporary/destination.name
                with zipfile.ZipFile(source) as bundle:
                    names=[n for n in bundle.namelist() if Path(n).name == model['zip_member']]
                    if len(names) != 1:
                        raise ValueError('无法定位预训练权重')
                    with bundle.open(names[0]) as entry,target.open('wb') as out:
                        shutil.copyfileobj(entry,out,length=1024*1024)
            else:
                target = source
            if sha256(target) != expected:
                raise ValueError(f'{destination.name} 的校验值不符')
            target.replace(destination)
        print(f'下载完成 {destination.name}',flush=True)

if __name__ == '__main__':
    main()
