"""Dataset checking, grouped splitting and identical image/mask geometry."""
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
USER = Path('/mnt/c/Users/spaho') if Path('/mnt/c/Users/spaho/Desktop').is_dir() else Path.home()
DEFAULT_ROOT = USER / 'Desktop' / 'embryos1,2,3,4'


def load_rows(root, manifest, verify=True):
    rows = json.loads(Path(manifest).read_text(encoding='utf-8'))['samples']
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    for row in rows:
        for field in ('image', 'mask'):
            path = (root / row[field]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                raise ValueError(f'Missing/invalid {field}: {path}')
            if verify and hashlib.sha256(path.read_bytes()).hexdigest() != row[field + '_sha256']:
                raise ValueError(f'File changed since mapping was checked: {path}')
        with Image.open(root / row['image']) as image, Image.open(root / row['mask']) as mask:
            labels = np.array(mask)
            if image.size != mask.size or labels.ndim != 2:
                raise ValueError(f'Image/mask size or channel mismatch: {row["id"]}')
            if not set(np.unique(labels).tolist()) <= {0, row['label_value']}:
                raise ValueError(f'Unexpected label values: {row["id"]}')
            if not (labels == row['label_value']).any():
                raise ValueError(f'Empty target mask: {row["id"]}')
    return rows


def make_splits(rows, seed=42):
    # Group = source folder + P code, as confirmed by the user.
    groups = sorted({r['group'] for r in rows})
    if len(groups) < 5:
        raise ValueError('Need at least five independent groups')
    rng = random.Random(seed)
    assigned = {}
    # Each source is represented in train, validation and test.
    for source in sorted({r['source'] for r in rows}):
        source_groups = sorted({r['group'] for r in rows if r['source']==source})
        if len(source_groups) < 5:
            raise ValueError('Need at least five embryo groups per source')
        rng.shuffle(source_groups)
        n_test = max(1, round(len(source_groups) * .15))
        n_val = max(1, round(len(source_groups) * .20))
        for i,g in enumerate(source_groups):
            if g in assigned:
                raise ValueError('Embryo group spans multiple sources; revise split strategy')
            assigned[g] = 'test' if i<n_test else 'val' if i<n_test+n_val else 'train'
    splits = {s: [r for r in rows if assigned[r['group']] == s] for s in ('train','val','test')}
    seen = {}
    for split, subset in splits.items():
        for r in subset:
            previous = seen.setdefault(r['image_sha256'], split)
            if previous != split:
                raise ValueError('Identical image appears in different splits')
    return splits


def letterbox(image, size, nearest=False):
    w,h = image.size
    scale = size / max(w,h)
    nw,nh = max(1, round(w*scale)), max(1, round(h*scale))
    left,top = (size-nw)//2, (size-nh)//2
    result = Image.new(image.mode, (size,size))
    result.paste(image.resize((nw,nh), Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR), (left,top))
    valid = np.zeros((size,size), dtype='float32')
    valid[top:top+nh,left:left+nw] = 1
    return result, valid, (left,top,nw,nh,w,h)


def prepare_pair(root, row, size, augment=False):
    with Image.open(root/row['image']) as im:
        image,valid,_ = letterbox(im.convert('RGB'), size)
    with Image.open(root/row['mask']) as im:
        binary = Image.fromarray((np.array(im) == row['label_value']).astype('uint8'))
        mask,_,_ = letterbox(binary,size,nearest=True)
    x = np.array(image,dtype='float32') / 255.0
    y = np.array(mask,dtype='float32')
    if not y.any():
        raise ValueError(f'Thin mask disappeared at size {size}: {row["id"]}. Increase --size.')
    if augment:
        if random.random() < .5:
            x,y,valid = np.flip(x,1),np.flip(y,1),np.flip(valid,1)
        if random.random() < .5:
            x,y,valid = np.flip(x,0),np.flip(y,0),np.flip(valid,0)
        # Vary illumination/contrast so predictions are less tied to one microscope.
        x = np.clip(x * random.uniform(.85,1.15),0,1)
        if random.random() < .8:
            contrast = random.uniform(.75,1.30)
            x = np.clip((x - .5) * contrast + .5, 0, 1)
        if random.random() < .3:
            gamma = random.uniform(.80,1.25)
            x = np.clip(x ** gamma, 0, 1)
        if random.random() < .25:
            noise = np.random.normal(0, .015, x.shape).astype('float32')
            x = np.clip(x + noise, 0, 1)
    # Fixed ImageNet normalization, shared by training, evaluation and prediction.
    x = (x - np.array([.485,.456,.406],dtype='float32')) / np.array([.229,.224,.225],dtype='float32')
    return np.ascontiguousarray(x.transpose(2,0,1)), np.ascontiguousarray(y[None]), np.ascontiguousarray(valid[None])
