import torch
import numpy as np
import torchvision
from torch import nn
import os
from tqdm import tqdm
import pdb
from torch.utils.data.sampler import SubsetRandomSampler

import sys
sys.path.append('../')
from datasets.pizza3d import Pizza3DDataset, denormalize_view_label
from common import get_lr, infinite_loader, save_captioned_image

def save_regressor(batch_idx, model, optimizer, args, ckpt_path):
    print(f'save model to {ckpt_path}')
    ckpt = {
        'args': args,
        'batch_idx': batch_idx,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(ckpt, ckpt_path)

def create_regressor(args, device='cuda'):
    if args.model == 'resnet50':
        model = torchvision.models.resnet50(pretrained=True)
    elif args.model == 'resnext101_32x8d':
        model = torchvision.models.resnext101_32x8d(pretrained=True)
    else:
        model = torchvision.models.vgg16(pretrained=True)
    
    num_feat = model.fc.in_features
    model.fc = nn.Linear(num_feat, 4) # 4 --> 4 + num_possible_shapes
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    return model, optimizer

def load_regressor(ckpt_path, device='cuda'):
    print(f'load model from {ckpt_path}')
    assert os.path.exists(ckpt_path)
    ckpt = torch.load(ckpt_path)
    ckpt_args = ckpt['args']
    batch_idx = ckpt['batch_idx']
    model, optimizer = create_regressor(ckpt_args, device)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    return ckpt_args, batch_idx, model, optimizer

if __name__ == '__main__':
    import argparse

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--dataset', default='pizza3d', choices=['pizza3d'])
        parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
        parser.add_argument('--ckpt_path', type=str, default='')
        parser.add_argument('--model', default='resnext101_32x8d', choices=['resnet50', 'resnext101_32x8d'])
        parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        parser.add_argument('--num_batches', type=int, default=5000, help='number of epochs to train for')
        parser.add_argument('--seed', default=8, type=int, help='manual seed')
        parser.add_argument('--wandb', default=1, type=int, choices=[0,1])
        return parser

    args = get_parser().parse_args()

    # ****** For Reproductability *******
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True
    device = args.device

    # ****** dataset *******
    if args.dataset == 'pizza3d':
        dataset = Pizza3DDataset(size=224)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.25 * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        print(len(train_indices), len(val_indices))
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, num_workers=16, drop_last=True)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=16, drop_last=True)
        print(len(train_loader), len(val_loader))
    else:
        raise Exception('Dataset not supported!')
    
    # ****** model *******
    if args.ckpt_path:
        _, batch, model, optimizer = load_regressor(args.ckpt_path)
        batch_start = batch + 1
    else:
        model, optimizer = create_regressor(args)
        batch_start = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')
    
    if device == 'cuda':
        model = nn.DataParallel(model)
        saved_model = model.module
    else:
        saved_model = model

    # setup loss criterion
    criterion = torch.nn.MSELoss()

    if args.wandb:
        import wandb
        project_name = 'MPG_BMVC21_view_regressor'
        wandb.init(project=project_name, config=args)
        wandb.config.update(args)
        save_dir = os.path.join(f'runs/{args.dataset}', wandb.run.id)
    else:
        from datetime import datetime
        dateTimeObj = datetime.now()
        time_stamp = dateTimeObj.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(os.path.dirname(__file__), f'runs/{args.dataset}', time_stamp)

    img_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(img_save_dir, exist_ok=True)

    # ****** train *******
    pbar = tqdm(range(batch_start, batch_start+args.num_batches), smoothing=0.1)
    train_loader = infinite_loader(train_loader)

    for batch_idx in pbar:
        img, tgt  = next(train_loader)
        view_point = tgt['view_label']
        img, view_point = img.to(device), view_point.to(device)
        output = model(img)
        loss = criterion(output, view_point)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx%500 == 0 or batch_idx==(batch_start+args.num_batches-1):
            ckpt_path = os.path.join(save_dir, f'{batch_idx:>08d}.ckpt')
            save_regressor(batch_idx, saved_model, optimizer, args, ckpt_path)
        
        if (batch_idx)%100 == 0:
            model.eval()
            crit_val = torch.nn.MSELoss(reduction='none')
            all_losses = [0.0, 0.0, 0.0, 0.0]
            with torch.no_grad():
                batch_val = 0
                count = 0
                for img, tgt in tqdm(val_loader):
                    view_point = tgt['view_label']
                    img, view_point = img.to(device), view_point.to(device)
                    output = model(img)
                    losses = crit_val(output, view_point)
                    angle_loss = losses[:,0].sum()
                    scale_loss = losses[:,1].sum()
                    dx_loss = losses[:,2].sum()
                    dy_loss = losses[:,3].sum()
                    loss = losses.sum()
                    bs = img.shape[0]
                    all_losses[0]+= (angle_loss)
                    all_losses[1]+= (scale_loss)
                    all_losses[2]+= (dx_loss)
                    all_losses[3]+= (dy_loss)

                    if batch_val < 3: # save the first three batches as illustration
                        view_point = denormalize_view_label(view_point.cpu())
                        output = denormalize_view_label(output.cpu())
                        caption = []
                        for i in range(bs):
                            angle = view_point[i][0]
                            scale = view_point[i][1]
                            dx = view_point[i][2]
                            dy = view_point[i][3]
                            angle_ = output[i][0]
                            scale_ = output[i][1]
                            dx_ = output[i][2]
                            dy_ = output[i][3]
                            cap = f'T:a={angle:.1f}|s={scale:.1f}|dx={dx:.1f}|dy={dy:.1f}\nP:a={angle_:.1f}|s={scale_:.1f}|dx={dx_:.1f}|dy={dy_:.1f}'
                            caption.append(cap)
                        fp = f'{img_save_dir}/{batch_idx:>08d}_{batch_val:>08d}.png'
                        save_captioned_image(caption, img, fp, font=12)
                        
                    batch_val += 1
                

            for i in range(len(all_losses)):
                all_losses[i] /= len(val_loader)*args.batch_size
            running_loss = sum(all_losses)
            scheduler.step(running_loss)
            log = {
                'loss': running_loss,
                'angle_loss':all_losses[0],
                'scale_loss':all_losses[1],
                'dx_loss':all_losses[2],
                'dy_loss':all_losses[3],
                'batch_idx': batch_idx,
                'lr': get_lr(optimizer),
            }
            if args.wandb:
                wandb.log(log)
            model.train()

    # save model
    if args.wandb:
        wandb.join()