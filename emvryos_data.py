"""Read this dataset's MATLAB v5 gTruth path lists without running MATLAB.

Supports the inspected image-only exports. Unexpected layouts fail explicitly.
Never pair files by alphabetic order or include session/alternative masks blindly.
"""
import hashlib
import re
import struct
import zlib
from pathlib import Path, PureWindowsPath

from PIL import Image


def matlab_pairs(path):
    data = path.read_bytes()
    if not data.startswith(b'MATLAB 5.0 MAT-file') or data[126:128] != b'IM':
        raise ValueError(f'Unsupported MATLAB format: {path}')
    chunks, pos = [], 128
    while pos + 8 <= len(data):
        kind, length = struct.unpack_from('<II', data, pos)
        payload = data[pos + 8:pos + 8 + length]
        if len(payload) != length:
            raise ValueError(f'Truncated MATLAB file: {path}')
        if kind == 15:
            chunks.append(zlib.decompress(payload))
            pos += 8 + length  # compressed MATLAB elements are not padded
        else:
            if kind == 14:
                chunks.append(payload)
            pos += 8 + ((length + 7) // 8) * 8
    raw = b'\n'.join(chunks)
    if b'groundTruth' not in raw:
        raise ValueError(f'No groundTruth object: {path}')
    strings = [s.decode('ascii') for s in re.findall(rb'[\x20-\x7e]{4,}', raw)]
    paths = [s for s in strings if re.match(r'^[A-Za-z]:\\', s) and s.lower().endswith('.png')]
    images = [s for s in paths if re.fullmatch(r'P\d+ \d+h .+\.png', PureWindowsPath(s).name)]
    masks = [s for s in paths if re.fullmatch(r'Label_\d+\.png', PureWindowsPath(s).name)]
    if not images or len(images) != len(masks) or paths != images + masks:
        raise ValueError(f'Unsupported/ambiguous gTruth path layout: {path}')
    if len(set(images)) != len(images) or len(set(masks)) != len(masks):
        raise ValueError(f'Duplicate paths in gTruth: {path}')
    return list(zip(images, masks))


def resolve_export_path(stored, candidates, root):
    """Match longest path suffix after moving a Windows MATLAB dataset."""
    wanted = [s.lower() for s in PureWindowsPath(stored).parts]
    scored = []
    for candidate in candidates:
        parts = [s.lower() for s in candidate.relative_to(root).parts]
        score = 0
        for a, b in zip(reversed(wanted), reversed(parts)):
            if a != b:
                break
            score += 1
        if score:
            scored.append((score, candidate))
    if not scored:
        raise ValueError(f'Missing file referenced by MATLAB: {stored}')
    best = max(s for s, _ in scored)
    matches = [p for s, p in scored if s == best]
    if len(matches) != 1:
        raise ValueError(f'Ambiguous MATLAB path {stored}: {matches}')
    return matches[0]


def discover_rows(root, include_contour_source=True):
    root = Path(root).resolve()
    exports = sorted(root.rglob('gTruth.mat'))
    if not exports:
        raise ValueError(f'No gTruth.mat exports found under {root}')
    rows, sources = [], set()
    for export in exports:
        relative = export.relative_to(root)
        source_match = re.fullmatch(r'embrios([1-4])', relative.parts[0])
        if not source_match:
            raise ValueError(f'Unexpected source directory: {export}')
        source = 'embryos' + source_match.group(1)
        if source in sources:
            raise ValueError(f'Multiple gTruth exports for {source}; select one explicitly')
        sources.add(source)
        source_root = root / relative.parts[0]
        # MATLAB session copies are never additional training samples.
        candidates = [p for p in source_root.rglob('*.png')
                      if not any(part.startswith('.') for part in p.relative_to(root).parts)]
        pairs = matlab_pairs(export)
        for index, (stored_image, stored_mask) in enumerate(pairs, 1):
            image = resolve_export_path(stored_image, candidates, root)
            mask = resolve_export_path(stored_mask, candidates, root)
            match = re.fullmatch(r'(P\d+) (\d+)h .+\.png', image.name)
            with Image.open(image) as im:
                width, height = im.size
            rows.append(dict(id=f'{source}_{index:03d}', source=source,
                             image=image.relative_to(root).as_posix(),
                             mask=mask.relative_to(root).as_posix(),
                             group=f'{source}_{match[1]}', timepoint_hours=int(match[2]),
                             width=width, height=height, label_value=1,
                             image_sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                             mask_sha256=hashlib.sha256(mask.read_bytes()).hexdigest(),
                             mapping_source=relative.as_posix(), mapping_row=index,
                             annotation_review='contour_only' if source=='embryos3' else 'zona_area_candidate'))
    excluded = [r for r in rows if r['source']=='embryos3']
    print(f'MATLAB exports: {len(exports)}; mapped image/mask pairs: {len(rows)}', flush=True)
    if excluded and not include_contour_source:
        print(f'Excluded {len(excluded)} embrios3 contour masks: zona area is the target. '
              'These require corrected area annotations.', flush=True)
        rows = [r for r in rows if r['source']!='embryos3']
    elif excluded:
        print('WARNING: including embrios3 contour masks alongside zona area masks.', flush=True)
    if not rows:
        raise ValueError('No zona area candidates remain')
    return rows
