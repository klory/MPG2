import yaml
from types import SimpleNamespace
import argparse
import sys
sys.path.append('../')
from common import ROOT

dataset_args = ['classifier', 'view_regressor', 'inception']
model_args = ['ckpt_dir', 'ckpt_path']


def load_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default=f'{ROOT}/metrics/configs/default.yaml')
    parser.add_argument('--dataset', type=str, default=f'pizza10')
    parser.add_argument('--model', type=str, default=f'mpg')
    parser.add_argument('--sweep', type=int, default=1, choices=[0,1])
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--truncation', type=float, default=1.0)
    parser.add_argument('--truncation_mean', type=int, default=4096)
    parser.add_argument('--n_sample', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    args = parser.parse_args()
    with open(args.config) as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    for p in dataset_args:
        setattr(args, p, data['datasets'][args.dataset][p])
    for p in model_args:
        setattr(args, p, data['models'][args.model][p])
    return args


if __name__ == '__main__':
    import pdb
    args = load_args()
    print(args)