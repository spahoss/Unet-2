"""Age selection and prediction compatibility; no torch dependency."""
import re
from pathlib import Path

AGES = (25, 69, 117)


def age_of(row):
    values = []
    if 'timepoint_hours' in row:
        value = row['timepoint_hours']
        if isinstance(value, bool) or str(value) not in {'25', '69', '117'}:
            raise ValueError(f'Invalid timepoint_hours: {row["id"]}: {value}')
        values.append(int(value))
    for field in ('original_image', 'image'):
        name = str(row.get(field, '')).replace('\\', '/').rsplit('/', 1)[-1]
        match = re.search(r'\b(\d+)h\b', name)
        if match:
            values.append(int(match[1]))
    if not values or len(set(values)) != 1 or values[0] not in AGES:
        raise ValueError(f'Missing, conflicting or unsupported age: {row["id"]}')
    return values[0]


def select_age(rows, age):
    if age is None:
        return rows
    selected = [dict(r, timepoint_hours=age) for r in rows if age_of(r)==age]
    if not selected:
        raise ValueError(f'No images for {age}h')
    return selected


def prediction_age(config, requested, image):
    trained = config.get('age')
    if requested is not None and trained != requested:
        raise ValueError(f'Requested {requested}h, but checkpoint age is {trained!r}. '
                         'Use the checkpoint trained with the corresponding --age.')
    effective = trained if requested is None else requested
    match = re.search(r'\b(\d+)h\b', Path(image).name)
    if effective is not None and match and int(match[1]) != effective:
        raise ValueError(f'Image filename indicates {match[1]}h, checkpoint is for {effective}h')
    return effective
