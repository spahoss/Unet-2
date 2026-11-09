"""Copy verified image/mask pairs into data/train, data/val and data/test.

python prepare_data.py
Original files are preserved. Existing destination folders are never overwritten.
"""
import argparse
import json
import shutil
from pathlib import Path
from data_utils import HERE, DEFAULT_ROOT, load_rows, make_splits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--manifest',type=Path,default=None)
    parser.add_argument('--exclude-contour-source',action='store_true')
    parser.add_argument('--destination',type=Path,default=HERE/'data_emvryos')
    parser.add_argument('--seed',type=int,default=42)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error(f'Destination already exists: {args.destination}. Use the existing data or choose another --destination.')
    rows = load_rows(args.source,args.manifest,include_contour_source=not args.exclude_contour_source)
    splits = make_splits(rows,args.seed)
    args.destination.mkdir(parents=True,exist_ok=False)
    prepared = []
    for split, subset in splits.items():
        for kind in ('images','masks'):
            (args.destination/split/kind).mkdir(parents=True,exist_ok=False)
        for row in subset:
            item = dict(row,split=split,original_image=row['image'],original_mask=row['mask'])
            for field,kind in (('image','images'),('mask','masks')):
                # Matching unique file stems make image/mask pairs easy to inspect.
                relative = Path(split)/kind/(row['id']+'.png')
                shutil.copy2(args.source/row[field],args.destination/relative)
                item[field] = relative.as_posix()
            prepared.append(item)
        print(f'{split}: {len(subset)} image/mask pairs')
    manifest = args.destination/'dataset.json'
    manifest.write_text(json.dumps(dict(samples=prepared,seed=args.seed,
        grouping='source folder + P code; all embryo timepoints stay together'),indent=2),encoding='utf-8')
    verified = load_rows(args.destination,manifest)
    make_splits(verified,args.seed)
    print(f'Copied and verified {len(verified)} pairs: {args.destination}')


if __name__=='__main__':
    main()
