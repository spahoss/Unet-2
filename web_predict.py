"""Upload validation and isolated prediction results for the Streamlit UI."""
import io
import json
import tempfile
import threading
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

from age_utils import prediction_age

MAX_BYTES = 20 * 1024 * 1024
MAX_PIXELS = 12_000_000


def decode_upload(payload, mask=False):
    if not payload or len(payload)>MAX_BYTES:
        raise ValueError('Το αρχείο πρέπει να έχει μέγεθος έως 20 MB.')
    with Image.open(io.BytesIO(payload)) as image:
        if image.width*image.height>MAX_PIXELS:
            raise ValueError('Η εικόνα πρέπει να έχει έως 12 εκατομμύρια pixel.')
        image.load()
        if not mask:
            return image.convert('RGB')
        labels=np.array(image)
        if labels.ndim!=2:
            raise ValueError('Η σωστή μάσκα πρέπει να έχει ένα κανάλι.')
        values=set(np.unique(labels).tolist())
        if values <= {0,1}:
            binary=labels.astype('uint8')
        elif values <= {0,255}:
            binary=(labels==255).astype('uint8')
        else:
            raise ValueError('Η μάσκα πρέπει να περιέχει μόνο 0/1 ή 0/255.')
        return Image.fromarray(binary)


def validate_config(config, age, filename):
    prediction_age(config,age,filename)
    if config.get('in_channels')!=3 or config.get('normalization')!='imagenet':
        raise ValueError('Η εφαρμογή υποστηρίζει τα μοντέλα U-Net/ResNet του παρόντος project με RGB και ImageNet normalization.')
    if not isinstance(config.get('encoder'),str) or not isinstance(config.get('size'),int):
        raise ValueError('Λείπουν οι απαραίτητες ρυθμίσεις από το checkpoint.')
    if config['size']<64 or config['size']>2048 or config['size']%32:
        raise ValueError('Μη υποστηριζόμενη ανάλυση εισόδου μοντέλου.')


def load_model(checkpoint_path, device_name, age):
    import torch
    import segmentation_models_pytorch as smp
    if device_name=='cuda' and not torch.cuda.is_available():
        raise ValueError('Η GPU δεν είναι διαθέσιμη. Επίλεξε CPU ή έλεγξε την εγκατάσταση CUDA.')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    config=checkpoint['config']
    validate_config(config,age,'uploaded_image.png')
    model=smp.Unet(encoder_name=config['encoder'],encoder_weights=None,
                   in_channels=3,classes=1)
    model.load_state_dict(checkpoint['model_state'])
    device=torch.device(device_name)
    model=model.to(device).eval()
    return model,device,config,threading.Lock()


def run_prediction(bundle, image_bytes, filename, age, mask_bytes=None,
                   threshold=.5, checkpoint_name='', predictor=None):
    model,device,config,lock=bundle
    validate_config(config,age,filename)
    if not 0<threshold<1:
        raise ValueError('Το όριο πιθανότητας πρέπει να είναι μεταξύ 0 και 1.')
    image=decode_upload(image_bytes)
    mask=decode_upload(mask_bytes,mask=True) if mask_bytes is not None else None
    if mask is not None and mask.size!=image.size:
        raise ValueError('Η εικόνα και η σωστή μάσκα πρέπει να έχουν τις ίδιες διαστάσεις.')
    if predictor is None:
        from predict import save_prediction
        predictor=save_prediction
    with tempfile.TemporaryDirectory(prefix='embryo_predict_') as folder:
        root=Path(folder)
        image_path=root/'input.png'
        image.save(image_path)
        mask_path=None
        if mask is not None:
            mask_path=root/'reference.png'
            mask.save(mask_path)
        destination=root/'result'
        with lock:
            metrics=predictor(model,device,image_path,config['size'],destination,mask_path,threshold,0.)
        files={p.name:p.read_bytes() for p in destination.iterdir() if p.is_file()}
    metadata=dict(age_hours=age,threshold=threshold,checkpoint=checkpoint_name,
                  image_size=list(image.size),metrics=metrics)
    files['prediction_info.json']=json.dumps(metadata,indent=2,ensure_ascii=False).encode('utf-8')
    archive=io.BytesIO()
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as zip_file:
        for name,payload in files.items():
            zip_file.writestr(name,payload)
    return dict(files=files,archive=archive.getvalue(),metrics=metrics,metadata=metadata)
