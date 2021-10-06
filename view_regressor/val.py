import torch
import numpy as np
import torchvision
from torch import nn
import os
from tqdm import tqdm
import wandb
import pdb
from torch.utils.data.sampler import SubsetRandomSampler

import sys
sys.path.append('../')
from datasets.pizza3d import Pizza3DDataset, denormalize_view_label
from view_regressor.train import load_regressor
from common import requires_grad, infinite_loader

if __name__ == '__main__':
    import argparse

    def get_parser():
        parser = argparse.ArgumentParser()
        parser.add_argument('--ckpt_path', type=str, default='runs/pizza3d/1ab8hru7/00004999.ckpt')
        parser.add_argument('--seed', default=8, type=int, help='manual seed')
        parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
        parser.add_argument('--dataset', default='pizza3d', choices=['pizza3d', 'labeledPizza10Subset'])
        parser.add_argument('--num_samples', type=int, default=1000, help='number of samples')
        parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
        return parser

    args = get_parser().parse_args()
    device = args.device

    # ****** For Reproductability *******
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # ****** model *******
    ckpt_args, _, model, _ = load_regressor(args.ckpt_path)
    model.to(device).eval()
    requires_grad(model, False)


    # ****** dataset *******
    if args.dataset == 'pizza3d':
        dataset = Pizza3DDataset(size=224)
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(0.25 * dataset_size))
        np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]
        valid_sampler = SubsetRandomSampler(val_indices)
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, sampler=valid_sampler, num_workers=16, drop_last=False)
    elif args.dataset == 'labeledPizza10Subset':
        from datasets.pizza10 import LabeledPizza10Subset
        dataset = LabeledPizza10Subset()
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=16, drop_last=False)
    
    # ****** validate *******
    labels_raw = []
    outputs_raw = []
    loader = infinite_loader(val_loader)
    args.num_samples = min(args.num_samples, len(dataset))
    for _ in tqdm(range(args.num_samples//args.batch_size+1)):
        img, tgt = next(loader)
        img = img.to(device)
        label = tgt['view_label'].to(device)
        output = model(img)
        labels_raw.append(label.cpu())
        outputs_raw.append(output.cpu())
    
    labels_raw = torch.cat(labels_raw, dim=0)
    outputs_raw = torch.cat(outputs_raw, dim=0)
    print(labels_raw.shape, outputs_raw.shape)
    labels = denormalize_view_label(labels_raw)
    preds = denormalize_view_label(outputs_raw)
    print('mean Abosolute Error')
    print(abs(labels - preds).mean(dim=0))
    print(abs(labels - preds).std(dim=0))
            
