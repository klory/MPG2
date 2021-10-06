import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from torchnet import meter

import pdb
import os
import csv
from glob import glob
from torch.nn import functional as F
from matplotlib import pyplot as plt
import sys
sys.path.append('../')
from common import requires_grad
from ingr_classifier.train import load_classifier


def compute_mAP(args, ingr_classifier):
    print(f'\nworking on {args.ckpt_path}')
    batch_generator = BatchGenerator(args)

    running_output = []
    running_label = []
    with torch.no_grad():
        for _ in tqdm(range(1000//args.batch_size)):
            fake_img, binary_label = batch_generator.generate_mAP()
            fake_img = normalize(resize(fake_img, size=224))
            output = ingr_classifier(fake_img)
            running_output.append(output)
            running_label.append(binary_label)
    
    running_output = torch.cat(running_output, dim=0)
    running_label = torch.cat(running_label, dim=0)
    
    mtr = meter.APMeter()
    mtr.add(running_output, running_label)
    APs = mtr.value()
    mAP = APs.mean().item() # mean average precision
    return mAP

if __name__ == '__main__':
    from common import load_args, normalize, resize
    args = load_args()

    # assertations
    assert 'ckpt_dir' in args.__dict__
    assert 'classifier' in args.__dict__
    assert 'device' in args.__dict__
    assert 'batch_size' in args.__dict__
    if 'mpg/' in args.ckpt_dir:
        from mpg.generate_batch import BatchGenerator
    
    device = args.device
    _, _, classifier, _ = load_classifier(args.classifier)
    classifier = classifier.eval().to(device)
    requires_grad(classifier, False)

    if not args.sweep:
        mAP = compute_mAP(args, classifier)
        print(f'mAP={mAP:.4f}')
        sys.exit(0)

    filename = os.path.join(args.ckpt_dir, 'mAP.csv')
    # load values that are already computed
    computed_iterations = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                computed_iterations += [row[0]]
    
    # prepare to write
    f = open(filename, mode='a')
    writer = csv.writer(f, delimiter=',')

    # find checkpoints
    ckpt_paths = glob(os.path.join(args.ckpt_dir, '*.ckpt')) + glob(os.path.join(args.ckpt_dir, '*.pt'))+glob(os.path.join(args.ckpt_dir, '*.pth'))
    iterations = [os.path.basename(ckpt_path).split('.')[0] for ckpt_path in ckpt_paths]
    ckpt_paths = sorted(ckpt_paths)
    print('records:', iterations)
    print('computed_iterations:', computed_iterations)
    for ckpt_path in ckpt_paths:
        iteration = os.path.basename(ckpt_path).split('.')[0]
        if iteration in computed_iterations:
            print('already computed')
            continue
        
        args.ckpt_path = ckpt_path
        mAP = compute_mAP(args, classifier)
        print(f'{iteration}, mAP={mAP:.4f}')
        writer.writerow([iteration, mAP])

        # # compute confusion matrix
        # if 'pizzaGANdata' == args.dataset:
        #     from sklearn.metrics import multilabel_confusion_matrix
        #     y_true = running_label.cpu().numpy()
        #     y_pred = (running_output>0.5).cpu().numpy()
        #     mats = multilabel_confusion_matrix(y_true, y_pred)
        #     for cat, mat, ap in zip(categories, mats, APs):
        #         print(cat)
        #         print(ap.item())
        #         print(mat)
        #         print()

    
    f.close()
    mAPs = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            mAP = float(row[1])
            mAPs += [mAP]
    fig = plt.figure(figsize=(6,6))
    plt.plot(mAPs)
    plt.savefig(os.path.join(args.ckpt_dir, 'mAP.png'))