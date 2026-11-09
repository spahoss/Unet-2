"""Load best_model.pt and segment a new image without retraining."""
import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from data_utils import HERE, letterbox
from age_utils import AGES, prediction_age


def save_prediction(model,device,image_path,size,destination,mask_path=None,threshold=.5,border_ratio=0.0):
    import torch
    with Image.open(image_path) as im:
        original = im.convert('RGB')
    frame,_,geometry = letterbox(original,size)
    x = np.array(frame,dtype='float32')/255
    x = (x-np.array([.485,.456,.406],dtype='float32'))/np.array([.229,.224,.225],dtype='float32')
    x = torch.from_numpy(np.ascontiguousarray(x.transpose(2,0,1)))[None].to(device)
    model.eval()
    with torch.inference_mode():
        prob = model(x).sigmoid()[0,0].cpu().numpy()
    left,top,nw,nh,w,h = geometry
    prob = np.array(Image.fromarray(prob[top:top+nh,left:left+nw]).resize((w,h),Image.Resampling.BILINEAR))
    raw_binary = prob>=threshold
    binary = raw_binary.copy()
    if border_ratio > 0:
        margin_y = max(1, round(binary.shape[0] * border_ratio))
        margin_x = max(1, round(binary.shape[1] * border_ratio))
        binary[:margin_y, :] = False
        binary[-margin_y:, :] = False
        binary[:, :margin_x] = False
        binary[:, -margin_x:] = False
    destination.mkdir(parents=True,exist_ok=False)
    Image.fromarray(raw_binary.astype('uint8')*255).save(destination/'prediction_raw.png')
    mask = Image.fromarray(binary.astype('uint8')*255)
    mask.save(destination/'prediction.png')
    Image.fromarray((prob*255).clip(0,255).astype('uint8')).save(destination/'probability.png')
    overlay = np.array(original,dtype='float32')
    overlay[binary] = .6*overlay[binary]+.4*np.array([255,40,100])
    overlay = Image.fromarray(overlay.astype('uint8'))
    overlay.save(destination/'overlay.png')
    panels = [('Input',original)]
    if mask_path is not None:
        with Image.open(mask_path) as lab:
            labels = np.array(lab)
            if labels.ndim != 2 or not set(np.unique(labels).tolist()) <= {0, 1}:
                raise ValueError('Reference mask must have one channel and label values 0 and 1')
            truth = labels == 1
            if truth.shape != binary.shape:
                raise ValueError('Image and reference mask must have identical dimensions')
            intersection = np.logical_and(binary, truth).sum()
            dice = float((2 * intersection + 1e-6) / (binary.sum() + truth.sum() + 1e-6))
            union = np.logical_or(binary, truth).sum()
            iou = float((intersection + 1e-6) / (union + 1e-6))
            accuracy = float(np.mean(binary == truth))
            panels.append(('Ground truth',Image.fromarray(truth.astype('uint8')*255).convert('RGB')))
    panels += [('Prediction',mask.convert('RGB')),('Overlay',overlay)]
    tw,th = 400,round(h*400/w)
    canvas = Image.new('RGB',(tw*len(panels),th+35),'white')
    draw = ImageDraw.Draw(canvas)
    for i,(label,panel) in enumerate(panels):
        draw.text((i*tw+8,10),label,fill='black')
        canvas.paste(panel.resize((tw,th)),(i*tw,35))
    canvas.save(destination/'comparison.png')
    if mask_path is not None:
        metrics = {'dice': dice, 'iou': iou, 'accuracy': accuracy,
                   'threshold': threshold, 'border_ratio': border_ratio,
                   'note': 'Original image resolution; final mask after optional border removal; accuracy is a fraction in [0,1].'}
        (destination/'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
        return metrics
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--age',type=int,choices=AGES,default=None,
                   help='Require a checkpoint trained for this age; otherwise infer from checkpoint')
    p.add_argument('--mask',type=Path,default=None,
                   help='Optional 0/1 ground-truth mask; prints Dice/IoU/pixel accuracy')
    p.add_argument('--threshold',type=float,default=.5,
                   help='Foreground probability threshold (default: 0.5)')
    p.add_argument('--border-ratio',type=float,default=0.0,
                   help='Clear predictions in this fraction of each image border (default: 0)')
    p.add_argument('--output',type=Path,default=None)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    a = p.parse_args()
    if not 0 < a.threshold < 1:
        p.error('--threshold must be between 0 and 1')
    if not 0 <= a.border_ratio < .5:
        p.error('--border-ratio must be between 0 and 0.5')
    import torch
    import segmentation_models_pytorch as smp
    torch.set_num_threads(4)
    device = torch.device(a.device)
    ckpt = torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    config = ckpt['config']
    try:
        selected_age = prediction_age(config, a.age, a.image)
    except ValueError as error:
        p.error(str(error))
    print(f'Model age: {selected_age if selected_age is not None else "all/unspecified"}',flush=True)
    model = smp.Unet(encoder_name=config['encoder'],encoder_weights=None,
                     in_channels=config['in_channels'],classes=1).to(device)
    model.load_state_dict(ckpt['model_state'])
    out = a.output or HERE/'predictions'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    metrics = save_prediction(model,device,a.image,config['size'],out,a.mask,a.threshold,a.border_ratio)
    print(f'Prediction saved: {out}')
    print(f'Foreground threshold: {a.threshold:.2f}')
    print(f'Border removal: {a.border_ratio:.2f}')
    if metrics is not None:
        print(f"Image Dice: {metrics['dice']:.4f} | Image IoU: {metrics['iou']:.4f} | Pixel accuracy: {metrics['accuracy']:.2%}")
    else:
        print('Accuracy, Dice and IoU unavailable: supply a ground-truth mask with --mask.')


if __name__=='__main__':
    main()
