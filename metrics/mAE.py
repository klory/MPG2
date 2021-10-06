import torch
from tqdm import tqdm
import os
from matplotlib import pyplot as plt
from glob import glob
import csv

    
@torch.no_grad()
def compute_mAE(args, view_regressor, scales=torch.tensor([75.0, 3.0, 112.0, 112.0])):
    print(f'\nworking on {args.ckpt_path}')
    batch_generator = BatchGenerator(args)
    labels_raw = []
    preds_raw = []
    for _ in tqdm(range(1000//args.batch_size)):
        img, view_label = batch_generator.generate_mAE()
        img = normalize(resize(img))
        output = view_regressor(img)
        labels_raw.append(view_label.cpu())
        preds_raw.append(output.cpu())
    labels_raw = torch.cat(labels_raw, dim=0)
    preds_raw = torch.cat(preds_raw, dim=0)
    labels = labels_raw * scales
    preds = preds_raw * scales
    return abs(labels - preds).mean(dim=0), abs(labels - preds).std(dim=0)

if __name__ == '__main__':
    import sys
    sys.path.append('../')
    from view_regressor.train import load_regressor
    from mpg.generate_batch import BatchGenerator
    from common import normalize, resize, requires_grad
    from metrics.utils import load_args
    
    args = load_args()
    device = args.device

    scales = torch.tensor([75.0, 3.0, 112.0, 112.0])
    _, _, view_regressor, _ = load_regressor(args.view_regressor)
    view_regressor = view_regressor.eval().to(device)
    requires_grad(view_regressor, False)

    if not args.sweep:
        means, stds = compute_mAE(args, view_regressor, scales=scales)
        print(means, stds)
        sys.exit(0)
    
    filename = os.path.join(args.ckpt_dir, 'mAE.csv')
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
            print(f'{iteration} is already computed')
            continue 
        args.ckpt_path = ckpt_path
        means, stds = compute_mAE(args, view_regressor, scales=scales)
        print(f'{iteration}, means = {means.tolist()}, stds={stds.tolist()}')
        writer.writerow([iteration, means.tolist(), stds.tolist()])
    f.close()