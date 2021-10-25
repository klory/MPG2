import pickle
import torch
import numpy as np
from scipy import linalg
from tqdm import tqdm
import pdb
import os
import csv
from glob import glob
from matplotlib import pyplot as plt

import sys
sys.path.append('../')
from metrics.calc_inception import load_patched_inception_v3

def compute_fid(args, inception, real_mean, real_cov):
    print(f'\nworking on {args.ckpt_path}')
    batch_generator = BatchGenerator(args)
    features = extract_features(batch_generator, inception, args)
    print(f'extracted {features.shape[0]} features')
    sample_mean = np.mean(features, 0)
    sample_cov = np.cov(features, rowvar=False)
    fid = calc_fid(sample_mean, sample_cov, real_mean, real_cov)
    return fid

@torch.no_grad()
def extract_features(batch_generator, inception, args):
    n_batches = args.n_sample // args.batch_size
    features = []
    for _ in tqdm(range(n_batches)):
        img = batch_generator.generate_fid()
        img = 2*(img-img.min())/(img.max()-img.min())-1
        feat = inception(img)[0].view(img.shape[0], -1)
        features.append(feat.to("cpu"))
    features = torch.cat(features, 0)
    return features.numpy()

def calc_fid(sample_mean, sample_cov, real_mean, real_cov, eps=1e-6):
    cov_sqrt, _ = linalg.sqrtm(sample_cov @ real_cov, disp=False)

    if not np.isfinite(cov_sqrt).all():
        print('product of cov matrices is singular')
        offset = np.eye(sample_cov.shape[0]) * eps
        cov_sqrt = linalg.sqrtm((sample_cov + offset) @ (real_cov + offset))

    if np.iscomplexobj(cov_sqrt):
        if not np.allclose(np.diagonal(cov_sqrt).imag, 0, atol=1e-3):
            m = np.max(np.abs(cov_sqrt.imag))

            raise ValueError(f'Imaginary component {m}')

        cov_sqrt = cov_sqrt.real

    mean_diff = sample_mean - real_mean
    mean_norm = mean_diff @ mean_diff

    trace = np.trace(sample_cov) + np.trace(real_cov) - 2 * np.trace(cov_sqrt)

    fid = mean_norm + trace

    return fid


if __name__ == '__main__':
    from metrics.utils import load_args
    from common import ROOT
    args = load_args()
    
    # assertations
    assert 'ckpt_dir' in args.__dict__
    assert 'inception' in args.__dict__
    assert 'view_regressor' in args.__dict__
    assert 'device' in args.__dict__
    assert 'n_sample' in args.__dict__
    assert 'batch_size' in args.__dict__


    if 'mpg/' in args.ckpt_dir:
        from mpg.generate_batch import BatchGenerator

    device = args.device
    args.ckpt_dir = ROOT / args.ckpt_dir
    args.view_regressor = ROOT / args.view_regressor

    print(f'load real image statistics from {args.inception}')
    with open(args.inception, 'rb') as f:
        embeds = pickle.load(f)
        real_mean = embeds['mean']
        real_cov = embeds['cov']
    # print(real_mean.shape, real_cov.shape)
    
    # load inception model
    inception = load_patched_inception_v3()
    inception = inception.eval().to(device)
    inception.requires_grad_(False)
    
    # *******************************
    # only run for one checkpoint
    # ********************************
    if not args.sweep:
        args.ckpt_path = ROOT / args.ckpt_path
        fid = compute_fid(args, inception, real_mean, real_cov)
        print(f'FID = {fid:.4f}')
        sys.exit(0)

    # *******************************
    # run for all checkpoints
    # ********************************
    filename = os.path.join(args.ckpt_dir, f'fid_{args.n_sample}.csv')
    # load values that are already computed_iterations
    computed_iterations = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                computed_iterations += [row[0]]
    
    # prepare to write
    f = open(filename, mode='a')
    writer = csv.writer(f, delimiter=',')
    
    ckpt_paths = glob(os.path.join(args.ckpt_dir, '*.ckpt')) + glob(os.path.join(args.ckpt_dir, '*.pt'))+glob(os.path.join(args.ckpt_dir, '*.pth'))
    ckpt_paths = sorted(ckpt_paths)
    iterations = [os.path.basename(ckpt_path).split('.')[0] for ckpt_path in ckpt_paths]
    print('records:', iterations)
    print('computed_iterations:', computed_iterations)
    for ckpt_path in ckpt_paths:
        iteration = os.path.basename(ckpt_path).split('.')[0]
        if iteration in computed_iterations:
            print(f'{iteration} is already computed_iterations')
            continue
        
        args.ckpt_path = ckpt_path
        fid = compute_fid(args, inception, real_mean, real_cov)
        print(f'FID = {fid}')
        writer.writerow([iteration, fid])

    f.close()
    fids = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            fid = float(row[1])
            fids += [fid]
    fig = plt.figure(figsize=(6,6))
    plt.plot(fids)
    plt.savefig(os.path.join(args.ckpt_dir, f'fid_{args.n_sample}.png'))
