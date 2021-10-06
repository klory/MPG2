import torch
import numpy as np
import torchvision
from torch import nn
import os
import sys
from tqdm import tqdm
from torchnet import meter
import pdb

import sys
sys.path.append('../')
from ingr_classifier.args import get_parser
from common import get_lr, infinite_loader, ROOT

def load_dataset(args):
    if args.dataset == 'pizza10':
        from datasets.pizza10 import Pizza10Dataset
        from datasets import utils
        train_set = Pizza10Dataset(part='train', transform=utils.resnet_transform_train)
        val_set = Pizza10Dataset(part='val', transform=utils.resnet_transform_val)
        categories = train_set.categories
    else:
        raise Exception(f'The dataset {args.dataset} is not supported')
    return train_set, val_set, categories

def save_classifier(batch_idx, model, optimizer, args, ckpt_path):
    print(f'save model to {ckpt_path}')
    ckpt = {
        'args': args,
        'batch_idx': batch_idx,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(ckpt, ckpt_path)

def create_classifier(args, device='cuda'):
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif args.model == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d(pretrained=True)
    else:
        raise Exception(f'The model {args.model} is not supported')
    
    num_feat = model.fc.in_features
    model.fc = nn.Linear(num_feat, getattr(args, 'num_categories', 10))
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model.to(device), optimizer

def load_classifier(ckpt_path, device='cuda'):
    print(f'load model from {ckpt_path}')
    assert os.path.exists(ckpt_path)
    ckpt = torch.load(ckpt_path)
    if 'batch' in ckpt:
        ckpt['batch_idx'] = ckpt.pop('batch')
    ckpt_args = ckpt['args']
    batch_idx = ckpt['batch_idx']
    model, optimizer = create_classifier(ckpt_args, device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt_args, batch_idx, model, optimizer

if __name__ == '__main__':
    args = get_parser().parse_args()

    if args.wandb:
        import wandb

    # ****** For Reproductability *******
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    device = 'cuda'

    # ****** dataset *******
    train_set, val_set, categories = load_dataset(args)
    print(categories)
    args.categories = categories
    args.num_categories = len(categories)
    print(len(train_set), len(val_set))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=2*args.batch_size, shuffle=False)
    print(len(train_loader), len(val_loader))

    # ****** model *******
    if args.ckpt_path:
        _, batch_idx, model, optimizer = load_classifier(args.ckpt_path, device)
        batch_start = batch_idx + 1
    else:
        model, optimizer = create_classifier(args, device)
        batch_start = 0
    
    if device == 'cuda':
        model = nn.DataParallel(model)
        saved_model = model.module
    else:
        saved_model = model
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max')

    # setup loss criterion
    # criterion = nn.BCEWithLogitsLoss(pos_weight=train_set.pos_weight).to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)

    if args.wandb:
        project_name = 'MPG_BMVC21_ingr_classifier'
        wandb.init(project=project_name, config=args)
        wandb.config.update(args)
        save_dir = os.path.join(f'runs', args.dataset, wandb.run.id)
    else:
        from datetime import datetime
        dateTimeObj = datetime.now()
        time_stamp = dateTimeObj.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(os.path.dirname(__file__), 'runs', args.dataset, time_stamp)
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    # ****** train *******
    loader = infinite_loader(train_loader)
    pbar = tqdm(range(batch_start, batch_start+args.num_batch), smoothing=0.3)
    for batch in pbar:
        # train phase
        img, tgt  = next(loader)
        label = tgt['ingr_label']
        img, label = img.to(device), label.to(device)
        output = model(img)
        loss = criterion(output, label.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%1000 == 0 or batch==(batch_start+args.num_batch-1):
            ckpt_path = os.path.join(save_dir, f'batch{batch}.ckpt')
            save_classifier(batch, saved_model, optimizer, args, ckpt_path)
        
        if (batch)%100 == 0:
            model.eval()
            running_loss = 0.0
            running_output = []
            running_label = []
            with torch.no_grad():
                batch_val = 0
                for img, tgt in val_loader:
                    label = tgt['ingr_label']
                    img, label = img.to(device), label.to(device)
                    bs = img.shape[0]
                    output = model(img)
                    loss = criterion(output, label.float()) 
                    running_loss += (loss*bs)
                    running_output.append(output)
                    running_label.append(label)
                    pbar.set_description(f'batch={batch_val:>3d}, loss={loss:.2f}')
                    batch_val += 1

            running_loss /= len(val_set)
            running_output = torch.cat(running_output, dim=0)
            running_label = torch.cat(running_label, dim=0)
            mtr = meter.APMeter()
            mtr.add(running_output, running_label)
            # call mtr.value() will return the APs (average precision) for each \
            # class, so the output will have shape [K]
            APs = mtr.value()
            mAP = APs.mean() # mean average precision
            scheduler.step(mAP)
            if args.wandb:
                log = {
                    'loss': running_loss,
                    'mAP': mAP,
                    'batch': batch,
                    'lr': get_lr(optimizer),
                }
                for idx, category in enumerate(args.categories):
                    log[f'AP_{category}'] = APs[idx]
                wandb.log(log)
            model.train()

    # save model
    wandb.join()