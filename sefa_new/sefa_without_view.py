"""SeFa."""

import os
import argparse
from tqdm import tqdm
import numpy as np

import torch
import sys
sys.path.append("../")
from common import ROOT
from sefa_new.train_cs import load_mpg
from sefa_new.models import parse_gan_type
from sefa_new.utils import to_tensor
from sefa_new.utils import postprocess
from sefa_new.utils import factorize_weight
from sefa_new.utils import HtmlPageVisualizer
from mpg_plusView.model_utils import requires_grad,infinite_loader,handle_viewpoint_img_input,mixing_noise
from torchvision import transforms
import math
# Set random seed.
np.random.seed(1)
torch.manual_seed(1)
def tile(a, dim, n_tile):
    a=a.cpu()
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)]))
    return torch.index_select(a, dim, order_index).cuda()
def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(
        description='Discover semantics from the pre-trained weight.')
    parser.add_argument('--ckpt_path', type=str,default=f'{ROOT}/mpg_plusView/runs/1ukmrcl9/530000.pt')
    parser.add_argument('--save_dir', type=str, default='results_withoutview',
                        help='Directory to save the visualization pages. '
                             '(default: %(default)s)')
    parser.add_argument('-L', '--layer_idx', type=str, default='all',
                        help='Indices of layers to interpret. '
                             '(default: %(default)s)')
    parser.add_argument('-N', '--num_samples', type=int, default=5,
                        help='Number of samples used for visualization. '
                             '(default: %(default)s)')
    parser.add_argument('-K', '--num_semantics', type=int, default=5,
                        help='Number of semantic boundaries corresponding to '
                             'the top-k eigen values. (default: %(default)s)')
    parser.add_argument('--start_distance', type=float, default=-5.0,
                        help='Start point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--end_distance', type=float, default=5.0,
                        help='Ending point for manipulation on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--step', type=int, default=11,
                        help='Manipulation step on each semantic. '
                             '(default: %(default)s)')
    parser.add_argument('--viz_size', type=int, default=256,
                        help='Size of images to visualize on the HTML page. '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_psi', type=float, default=0.7,
                        help='Psi factor used for truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--trunc_layers', type=int, default=8,
                        help='Number of layers to perform truncation. This is '
                             'particularly applicable to StyleGAN (v1/v2). '
                             '(default: %(default)s)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Seed for sampling. (default: %(default)s)')
    parser.add_argument('--gpu_id', type=str, default='0',
                        help='GPU(s) to use. (default: %(default)s)')
    parser.add_argument('--device', type=str, default='cuda')
                        
    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    os.makedirs(args.save_dir, exist_ok=True)
    device='cuda'
    # Factorize weights.

    ckpt_args, batch, label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim = load_mpg(args.ckpt_path)
    # generator =g_ema
    # don't use the first layer of convs
    affine_weight_dct = {}
    equalLinear = generator.__getattr__("conv1").__getattr__("conv").__getattr__("modulation")
    affine_weight_dct[0]=equalLinear.weight.data
    for i in range(1,generator.num_layers-1):
        equalLinear = generator.__getattr__("convs").__getattr__(str(i)).__getattr__("conv").__getattr__("modulation")
        affine_weight_dct[i]=equalLinear.weight.data
    print(affine_weight_dct.keys())

    # viewpoint_classifier = load_viewpoint_classifier()
    # loader = load_pizzaGANdataloader(batch=args.num_samples,shuffle=False)
    # loader=infinite_loader(loader)
    # real_img,data = next(loader)
    binary_label=torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
        [0., 0., 1., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
        [0., 0., 0., 0., 1., 0., 1., 0., 0., 0.],
        [1., 0., 1., 0., 0., 0., 0., 0., 0., 0.]])

    # print(binary_label)
    # real_img = real_img.to(device)
    # wrong_img = wrong_img.to(device)
    # binary_label = binary_label.to(device)
    # viewpoint_img = handle_viewpoint_img_input(real_img)
    # viewpoint_output = viewpoint_classifier(viewpoint_img)
    # viewpoint_label =torch.tensor([[0.1, 0.1, 0., 0.],
    #     [0.02, 0.1, 0.1, 0.1],
    #     [0.1, 0.2, 0.05, 0.05],
    #     [0.01, 0,  0.05, 0.05],
    #     [0.01, 0.2,  0.05,  0.05]]).to(device)

    requires_grad(label_encoder, False)
    requires_grad(generator, False)
    requires_grad(discriminator, True)

    noise = torch.randn(args.num_samples, 256).to(device)

    label_embedding = label_encoder(binary_label)

    gan_type = parse_gan_type(generator)
    layers, boundaries, values = factorize_weight(generator,affine_weight_dct, args.layer_idx)

    

    # Prepare codes.
    codes = None
    if gan_type == 'pggan':
        codes = generator.layer0.pixel_norm(codes)
    elif gan_type in ['stylegan2']:
        codes = generator.get_latent(label_embedding,noise,truncation=args.trunc_psi)
    codes=tile(codes,1,2)
    codes=codes[:,1:,:]
    print(codes.size())
    codes = codes.detach().cpu().numpy()

    # Generate visualization pages.
    distances = np.linspace(args.start_distance,args.end_distance, args.step)
    num_sam = args.num_samples
    num_sem = args.num_semantics
    vizer_1 = HtmlPageVisualizer(num_rows=num_sem * (num_sam + 1),
                                 num_cols=args.step + 1,
                                 viz_size=args.viz_size)
    vizer_2 = HtmlPageVisualizer(num_rows=num_sam * (num_sem + 1),
                                 num_cols=args.step + 1,
                                 viz_size=args.viz_size)

    headers = [''] + [f'Distance {d:.2f}' for d in distances]
    vizer_1.set_headers(headers)
    vizer_2.set_headers(headers)
    for sem_id in range(num_sem):
        value = values[sem_id]
        vizer_1.set_cell(sem_id * (num_sam + 1), 0,
                         text=f'Semantic {sem_id:03d}<br>({value:.3f})',
                         highlight=True)
        for sam_id in range(num_sam):
            vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, 0,
                             text=f'Sample {sam_id:03d}')
    for sam_id in range(num_sam):
        vizer_2.set_cell(sam_id * (num_sem + 1), 0,
                         text=f'Sample {sam_id:03d}',
                         highlight=True)
        for sem_id in range(num_sem):
            value = values[sem_id]
            vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, 0,
                             text=f'Semantic {sem_id:03d}<br>({value:.3f})')

    for sam_id in tqdm(range(num_sam), desc='Sample ', leave=False):
        code = codes[sam_id:sam_id + 1]
        for sem_id in tqdm(range(num_sem), desc='Semantic ', leave=False):
            boundary = boundaries[sem_id:sem_id + 1]
            for col_id, d in enumerate(distances, start=1):
                temp_code = code.copy()
                if gan_type == 'pggan':
                    temp_code += boundary * d
                    image = generator(to_tensor(temp_code))['image']
                elif gan_type in ['stylegan', 'stylegan2']:
                    temp_code[:, layers, :] += boundary * d
                    image,_= generator.synthesis(to_tensor(temp_code),randomize_noise=False)
                image = postprocess(image)[0]
                vizer_1.set_cell(sem_id * (num_sam + 1) + sam_id + 1, col_id,
                                 image=image)
                vizer_2.set_cell(sam_id * (num_sem + 1) + sem_id + 1, col_id,
                                 image=image)

    prefix = ('stylegan2'+f'N{num_sam}_K{num_sem}_L{args.layer_idx}_seed{args.seed}')
    vizer_1.save(os.path.join(args.save_dir, f'{prefix}_sample_first.html'))
    vizer_2.save(os.path.join(args.save_dir, f'{prefix}_semantic_first.html'))


if __name__ == '__main__':
    main()
