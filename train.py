"""Train a library U-Net on the supplied embryo image/mask pairs.

python train.py --check
python train.py --epochs 30 --device cpu
python train.py --epochs 1 --weights none --size 256 --run-name smoke
"""
import argparse
import csv
import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
from data_utils import HERE, DEFAULT_ROOT, load_rows, make_splits, prepare_pair
from age_utils import AGES, select_age


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root',type=Path,default=DEFAULT_ROOT)
    p.add_argument('--manifest',type=Path,default=None)
    p.add_argument('--age',type=int,choices=AGES,default=None,
                   help='Train only this age in hours; omitted trains all ages')
    p.add_argument('--exclude-contour-source',action='store_true',
                   help='Optionally exclude embrios3; default includes all 114 pairs as requested')
    p.add_argument('--check',action='store_true',help='Check all files and splits without training')
    p.add_argument('--epochs',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--size',type=int,default=512)
    p.add_argument('--lr',type=float,default=0.0001)
    p.add_argument('--weights',choices=['imagenet','none'],default='imagenet')
    p.add_argument('--device',choices=['cpu','cuda','auto'],default='cpu')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--patience',type=int,default=10)
    p.add_argument('--run-name',default=None)
    p.add_argument('--plot',action='store_true',help='Serve live training charts on localhost; HTML snapshots are always saved')
    p.add_argument('--summary',action='store_true',help='Print and save model layer/activation/parameter table before training')
    p.add_argument('--summary-only',action='store_true',help='Save the model layer table without training')
    args = p.parse_args()
    if args.manifest is None:
        args.manifest = args.data_root/'dataset.json' if (args.data_root/'dataset.json').is_file() else None
    if args.size < 64 or args.size % 32 or min(args.epochs,args.batch_size,args.patience) < 1 or args.lr <= 0:
        p.error('Positive epochs, batch-size, patience, lr required; size must be >=64 and divisible by 32')
    rows = load_rows(args.data_root,args.manifest,include_contour_source=not args.exclude_contour_source)
    rows = select_age(rows,args.age)
    # Separate reproducible random draws for each age-specific model.
    split_seed = args.seed if args.age is None else args.seed + args.age
    splits = make_splits(rows,split_seed)
    print('Verified images/masks:',len(rows),flush=True)
    for name, subset in splits.items():
        print(f'{name}: {len(subset)} images; groups {sorted({r["group"] for r in subset})}',flush=True)
    print(f'Age: {args.age if args.age is not None else "all"}; grouping: source folder + P code.',flush=True)
    print(f'Training seed: {args.seed}; split seed: {split_seed}. Same data + seeds reproduce the split. Prepared datasets retain their saved split.',flush=True)
    for r in rows:
        prepare_pair(args.data_root,r,args.size)
    if args.check:
        print('All pairs and resized masks checked. No training performed.')
        return

    import torch
    import segmentation_models_pytorch as smp
    from torch.utils.data import Dataset, DataLoader
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    if args.device == 'cpu':
        device = torch.device('cpu')
    else:
        available = torch.cuda.is_available()
        if args.device == 'cuda' and not available:
            raise RuntimeError('CUDA unavailable. Use --device cpu.')
        device = torch.device('cuda' if available else 'cpu')
    print('Device:',device,flush=True)

    class Embryos(Dataset):
        def __init__(self,subset,augment=False):
            self.rows,self.augment = subset,augment
        def __len__(self):
            return len(self.rows)
        def __getitem__(self,index):
            return tuple(torch.from_numpy(a) for a in prepare_pair(
                args.data_root,self.rows[index],args.size,self.augment))

    loaders = {s: DataLoader(Embryos(rs,s=='train'),batch_size=args.batch_size,
                            shuffle=s=='train',num_workers=0) for s,rs in splits.items()}
    # Complete U-Net architecture comes from the library; no custom U-Net layers.
    model = smp.Unet(encoder_name='resnet18',
                     encoder_weights='imagenet' if args.weights=='imagenet' else None,
                     in_channels=3,classes=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr)

    def loss_fn(logits,target,valid):
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits,target,reduction='none')
        bce = (bce*valid).sum() / valid.sum().clamp_min(1)
        prob = logits.sigmoid()*valid
        target = target*valid
        axes = (1,2,3)
        dice = (2*(prob*target).sum(axes)+1e-6)/(prob.sum(axes)+target.sum(axes)+1e-6)
        return bce + 1-dice.mean()

    def evaluate(loader):
        model.eval()
        losses,dices,ious,accuracies = [],[],[],[]
        with torch.inference_mode():
            for x,y,v in loader:
                x,y,v = x.to(device),y.to(device),v.to(device)
                logits = model(x)
                losses.extend([loss_fn(logits,y,v).item()]*len(x))
                pred = (logits.sigmoid()>=.5)&(v>0)
                truth = (y>0)&(v>0)
                axes = (1,2,3)
                intersection = (pred&truth).sum(axes).float()
                union = (pred|truth).sum(axes).float()
                total = pred.sum(axes)+truth.sum(axes)
                # Padding must not count as correctly predicted background.
                valid_pixels = v > 0
                correct = ((pred == truth) & valid_pixels).sum(axes).float()
                accuracy = correct / valid_pixels.sum(axes).clamp_min(1)
                accuracies.extend(accuracy.cpu().tolist())
                dices.extend(((2*intersection+1e-6)/(total+1e-6)).cpu().tolist())
                ious.extend(((intersection+1e-6)/(union+1e-6)).cpu().tolist())
        return dict(loss=float(np.mean(losses)),dice=float(np.mean(dices)),iou=float(np.mean(ious)),
                    accuracy=float(np.mean(accuracies)))

    name = args.run_name or ((f'zona_{args.age}h_' if args.age is not None else '') + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    if Path(name).name != name or name in ('.','..') or '/' in name or '\\' in name:
        p.error('--run-name must be a folder name, not a path')
    run = HERE/'runs'/name
    run.mkdir(parents=True,exist_ok=False)
    split_data = {s:[dict(r,split=s) for r in rs] for s,rs in splits.items()}
    (run/'split.json').write_text(json.dumps(split_data,indent=2),encoding='utf-8')
    config = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
    config.update(encoder='resnet18',in_channels=3,label_value=1,threshold=.5,split_seed=split_seed,
                  split_strategy='saved' if any('split' in r for r in rows) else 'random grouped per source and age',
                  normalization='imagenet',torch_version=str(torch.__version__),smp_version=str(smp.__version__),
                  metric_resolution=args.size,grouping='source folder + P code; user-confirmed')
    (run/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    if args.summary or args.summary_only:
        from model_summary import describe_model, save_model_summary
        sample = torch.zeros((1,3,args.size,args.size),device=device)
        save_model_summary(describe_model(model,sample),run)
        del sample
        if args.summary_only:
            print('Layer analysis completed. No training performed.',flush=True)
            return
    from training_progress import TrainingProgress
    progress = TrainingProgress(run, dict(run_name=name,age=args.age,epochs=args.epochs,
        batches_per_epoch=len(loaders['train']),max_iterations=args.epochs*len(loaders['train']),
        device=str(device),lr=args.lr,batch_size=args.batch_size,size=args.size), live=args.plot)
    # Always preserve progress and close the server, including interrupted runs.
    import atexit
    atexit.register(progress.on_exit)
    print(f'Saved charts: {run/"training_progress.html"}',flush=True)
    best,bad_epochs = -1.0,0
    with (run/'history.csv').open('w',newline='',encoding='utf-8') as file:
        writer = csv.DictWriter(file,fieldnames=['epoch','train_loss','train_accuracy','val_loss','val_dice','val_iou','val_accuracy'])
        writer.writeheader()
        for epoch in range(1,args.epochs+1):
            model.train()
            total_loss,total_accuracy,total_images = 0.,0.,0
            for batch,(x,y,v) in enumerate(loaders['train'],1):
                x,y,v = x.to(device),y.to(device),v.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = loss_fn(logits,y,v)
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite loss; stopping')
                loss.backward()
                optimizer.step()
                with torch.no_grad():
                    valid_pixels = v > 0
                    correct = (((logits.detach().sigmoid()>=.5)==(y>0)) & valid_pixels).sum((1,2,3)).float()
                    batch_accuracy = (correct/valid_pixels.sum((1,2,3)).clamp_min(1)).mean().item()
                progress.batch(epoch,(epoch-1)*len(loaders['train'])+batch,loss.item(),batch_accuracy)
                total_loss += loss.item()*len(x)
                total_accuracy += batch_accuracy*len(x)
                total_images += len(x)
                if batch==1 or batch%10==0:
                    print(f'Epoch {epoch}/{args.epochs} batch {batch}/{len(loaders["train"])} loss={loss.item():.4f}',flush=True)
            val = evaluate(loaders['val'])
            metrics = dict(epoch=epoch,train_loss=total_loss/total_images,train_accuracy=total_accuracy/total_images,
                           val_loss=val['loss'],val_dice=val['dice'],val_iou=val['iou'],val_accuracy=val['accuracy'])
            writer.writerow(metrics)
            file.flush()
            progress.epoch(metrics)
            print(f'Epoch {epoch}: train_loss={metrics["train_loss"]:.4f} val_dice={val["dice"]:.4f} val_accuracy={val["accuracy"]:.2%}',flush=True)
            if val['dice'] > best:
                best,bad_epochs = val['dice'],0
                # CPU tensors make the checkpoint usable on CPU or GPU.
                torch.save(dict(model_state={k:v.detach().cpu() for k,v in model.state_dict().items()},
                                config=config,epoch=epoch,val_dice=best),run/'best_model.pt')
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print('Early stopping: validation Dice stopped improving.',flush=True)
                    break

    progress.status('Training finished — evaluating test')
    checkpoint = torch.load(run/'best_model.pt',map_location=device,weights_only=True)
    model.load_state_dict(checkpoint['model_state'])
    # Test is evaluated ONCE after checkpoint selection using validation only.
    test = evaluate(loaders['test'])
    test.update(best_epoch=checkpoint['epoch'],images=len(splits['test']),age=args.age,
                note='Mean per-image metrics at resized input resolution; grouped by embryo.')
    (run/'test_metrics.json').write_text(json.dumps(test,indent=2),encoding='utf-8')
    print('Held-out test:',test,flush=True)
    from predict import save_prediction
    for row in splits['test']:
        save_prediction(model,device,args.data_root/row['image'],args.size,
                        run/'test_predictions'/row['id'],args.data_root/row['mask'])
    progress.status('Completed')
    progress.close()
    print(f'Completed. Model and results: {run}',flush=True)


if __name__=='__main__':
    main()
