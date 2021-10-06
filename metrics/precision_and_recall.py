import os
import torch
from glob import glob
import csv
from torchvision import models
from tqdm import tqdm
from common import load_args, resize, normalize
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from matplotlib import pyplot as plt
from torchvision.utils import save_image
import pdb

# by default,
# k=3
# num_images = 5000 

@torch.no_grad()
def extract_features(batch_generator, model, args):
    n_batches = args.n_sample // args.batch_size
    real_features = []
    fake_features = []

    # # BUG: zero recalls
    # save_dir = f'samples/{args.model_name}'
    # os.makedirs(save_dir, exist_ok=True)

    for batch_idx in tqdm(range(n_batches)):
        real, fake = batch_generator.generate_ssim()
        real = normalize(resize(real))
        fake = normalize(resize(fake))

        # # BUG: zero recalls
        # save_image(real, f'{save_dir}/batch{batch_idx}_real.jpg', nrow=8, padding=2, normalize=True)
        # save_image(fake, f'{save_dir}/batch{batch_idx}_fake.jpg', nrow=8, padding=2, normalize=True)

        real_feat = model(real)
        fake_feat = model(fake)
        real_features.append(real_feat.to("cpu"))
        fake_features.append(fake_feat.to("cpu"))
    real_features = torch.cat(real_features, 0)
    fake_features = torch.cat(fake_features, 0)
    
    real_features = real_features.view(real_features.shape[0],-1)
    fake_features = fake_features.view(fake_features.shape[0],-1)
    return real_features.numpy(), fake_features.numpy()

@torch.no_grad()
def extract_features_only_fake(batch_generator, model, args):
    n_batches = args.n_sample // args.batch_size
    fake_features = []
    for batch_idx in tqdm(range(n_batches)):
        fake = batch_generator.generate_fid()
        fake = normalize(resize(fake))
        fake_feat = model(fake)
        fake_features.append(fake_feat.to("cpu"))
    fake_features = torch.cat(fake_features, 0)
    fake_features = fake_features.view(fake_features.shape[0],-1)
    return fake_features.numpy()

def find_radius(features, k=3):
    dists = euclidean_distances(features)
    dists.sort(axis=1)
    return dists[:,k]

def compute_precision_and_recall(real_features, fake_features, k=3):
    # pdb.set_trace()
    real_radius = find_radius(real_features)
    fake_radius = find_radius(fake_features)
    dists = euclidean_distances(real_features, fake_features)

    # precision: how many fake are in the support or real images?
    tmp = np.expand_dims(real_radius, 1) - dists
    precision_flags = np.any(tmp>=0, axis=0)
    precision = sum(precision_flags)/len(precision_flags)

    # recall: how many real are in the support of fake images?
    tmp = np.expand_dims(fake_radius, 0) - dists
    recall_flags = np.any(tmp>=0, axis=1)
    recall = sum(recall_flags)/len(recall_flags)
    
    print(dists.shape, precision, recall)
    return precision, recall


if __name__ == '__main__':
    args = load_args()
    
    # assertations
    assert 'ckpt_dir' in args.__dict__
    assert 'device' in args.__dict__
    assert 'n_sample' in args.__dict__
    assert 'batch_size' in args.__dict__

    if 'mpg' in args.ckpt_dir:
        from mpg.generate_batch import BatchGenerator
    else:
        raise Exception('Unsupported model')

    device = args.device

    filename = os.path.join(args.ckpt_dir, f'precisions_and_recalls_{args.n_sample}.csv')
    # load values that are already computed
    computed = []
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                computed += [row[0]]

    # load vgg16
    extractor = models.vgg16(pretrained=True).eval().to(device)
    extractor.classifier = torch.nn.Sequential(*[extractor.classifier[i] for i in range(5)])

    # # load classifier
    # _, _, classifier, _ = load_classifier(args.classifier, device=device)
    # classifier.eval()
    # modules = list(classifier.children())[:-1]
    # extractor = nn.Sequential(*modules)
    
    # prepare to write
    f = open(filename, mode='a')
    writer = csv.writer(f, delimiter=',')

    ckpt_paths = glob(os.path.join(args.ckpt_dir, '*.ckpt')) + glob(os.path.join(args.ckpt_dir, '*.pt'))+glob(os.path.join(args.ckpt_dir, '*.pth'))
    ckpt_paths = sorted(ckpt_paths)[:10]
    print('records:', ckpt_paths)
    print('computed:', computed)
    real_features = None

    # BUG: Recalls are zero for baselines
    # args.model_name = 'StackGAN2'
    # ckpt_paths = ckpt_paths[2:3]

    # args.model_name = 'CookGAN'
    # ckpt_paths = ckpt_paths[3:4]

    # args.model_name = 'AttnGAN'
    # ckpt_paths = ckpt_paths[9:10]

    # args.model_name = 'MPG'
    # ckpt_paths = ckpt_paths[8:9]

    for ckpt_path in ckpt_paths:
        print()
        print(f'working on {ckpt_path}')
        iteration = os.path.basename(ckpt_path).split('.')[0]
        if iteration in computed:
            print('already computed')
            continue
        
        args.ckpt_path = ckpt_path
        batch_generator = BatchGenerator(args)

        if real_features is None:
            real_features, fake_features = extract_features(batch_generator, extractor, args)
        else:
            fake_features = extract_features_only_fake(batch_generator, extractor, args)

        print(f'extracted {real_features.shape[0]} features')
        precision, recall = compute_precision_and_recall(real_features, fake_features)
        
        print(f'{iteration}, precision={precision:.2f}, recall={recall:.2f}')
        writer.writerow([iteration, precision, recall])

        del fake_features
        torch.cuda.empty_cache()


    f.close()
    precisions = []
    recalls = []
    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            precision = float(row[1])
            precisions += [precision]
            recall = float(row[2])
            recalls += [recall]
    fig = plt.figure(figsize=(6,6))
    plt.plot(precisions, label='precision')
    plt.plot(recalls, label='recall')
    plt.legend()
    plt.savefig(os.path.join(args.ckpt_dir, f'pr_{args.n_sample}.png'))
