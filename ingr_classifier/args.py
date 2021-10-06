import argparse
import os
from common import ROOT

def get_parser():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--dataset', default='pizza10', choices=['pizza10'])
    
    parser.add_argument('--ckpt_path', default='')
    # parser.add_argument('--ckpt_path', default='runs/pizza10/1t4xrvwx/batch5000.ckpt')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--model', default='resnet50', choices=['resnet50', 'resnext101_32x8d'])
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--num_batch', type=int, default=10000, help='number of epochs to train for')
    parser.add_argument('--seed', default=8, type=int, help='manual seed')
    parser.add_argument('--wandb', default=1, type=int, choices=[0, 1])

    # For CelebA-HQ
    parser.add_argument('--folder', default='cartoon', choices=['cartoon', 'source'])
    parser.add_argument('--only_significant', default=1, type=int, choices=[0, 1])
    return parser
