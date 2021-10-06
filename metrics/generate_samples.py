import torch
from PIL import Image, ImageFont, ImageDraw
from torchvision.utils import save_image
from torchvision import transforms
import os
import pdb
import numpy as np
import sys
sys.path.append('../')
import common


def randomize_ingr_label(bs):
    ingr_label = torch.zeros(bs, 10)
    for i in range(bs):
        idxs = np.random.choice(10, np.random.randint(4), replace=False)
        ingr_label[i, idxs] = 1.0
    return ingr_label


if __name__ == '__main__':
    from common import load_args
    args = load_args()
    
    # assertations
    assert 'ckpt_path' in args.__dict__
    assert 'device' in args.__dict__
    assert 'batch_size' in args.__dict__
    assert 'truncation' in args.__dict__
    assert 'model_name' in args.__dict__

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    print('current file dir:', cur_dir)
    if 'stackgan2/' in args.ckpt_dir:
        from stackgan2.generate_batch import BatchGenerator
        os.chdir('../stackgan2/')
    elif 'AttnGAN/' in args.ckpt_dir:
        from AttnGAN.code.generate_batch_Attn import BatchGenerator
        os.chdir('../AttnGAN/code/')
    elif 'mpg/' in args.ckpt_dir:
        from mpg.generate_batch import BatchGenerator
        os.chdir('../mpg/')
    elif 'stylegan2_cat_z/' in args.ckpt_dir:
        from stylegan2_cat_z.generate_batch import BatchGenerator
        os.chdir('../stylegan2_cat_z/')
    elif 'mpg_plusView/' in args.ckpt_dir:
        from mpg_plusView.generate_batch import BatchGenerator
        os.chdir('../mpg_plusView/')

    device = args.device

    def set_seed(seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    seed = args.seed

    set_seed(seed)
    save_dir = f'outputs/seed={seed}'
    os.makedirs(save_dir, exist_ok=True)
    batch_generator = BatchGenerator(args)
    os.chdir(cur_dir)
    
    # ****************************************************************
    # Part 1
    # ****************************************************************

    print('generating images...')
    with torch.no_grad():
        txt, real, fake, label = batch_generator.generate_all()

    if 'mpg_plusView' in args.ckpt_dir:
        caption = []
        view_attrs = ['angle', 'scale', 'dx', 'dy']
        scales = torch.tensor([75.0, 3.0, 112.0, 112.0]).unsqueeze(0)
        view_label = (label[:, 10:].cpu()*scales).tolist()
        for one_txt, one_view_label in zip(txt, view_label):
            one_caption = one_txt
            one_view_value = '\n'.join([f'{view_attrs[i]} = {x:.2f}' for i,x in enumerate(one_view_label)])
            one_caption += '\n'
            one_caption += one_view_value
            caption.append(one_caption)
    else:
        caption = txt

    os.makedirs('outputs', exist_ok=True)
    fp = f'{save_dir}/{args.model_name}_trunc={args.truncation:.2f}.png'
    common.save_captioned_image(caption, fake, fp, font=15, opacity=0.2, color=(255,255,0), loc=(0,0), nrow=int(np.sqrt(args.batch_size)), pad_value=1)
    print(f'saved to {fp}')



    # ****************************************************************
    # Part 2
    # ****************************************************************

    # if 'mpg_plusView' in args.ckpt_dir:
    #     class Attr:
    #         ANGLE=0
    #         SCALE=1
    #         DX=2
    #         DY=3
        
    #     view_attrs = ['angle', 'scale', 'dx', 'dy']
        
    #     print(f'\ngenerate images from label')
    #     stats = torch.tensor([
    #         [0.16, 0.12],
    #         [0.81, 0.06],
    #         [0.01, 0.08],
    #         [-0.08, 0.08]
    #     ])

    #     full_ranges = torch.tensor([
    #         [0.0, 75.0],
    #         [1.0, 3.0],
    #         [-112.0, 112.0],
    #         [-112.0, 112.0]
    #     ])

    #     def create_ingr_label(nrow):
    #         ingr_label = torch.zeros(nrow, 10)
    #         for i in range(nrow):
    #             idxs = np.random.choice(10, np.random.randint(4), replace=False)
    #             ingr_label[i, idxs] = 1.0
    #         return ingr_label # [nrow, 10]

    #     def create_view_label(nrow, attr_idx, attr_val):
    #         view_label = stats[:, 0].t().repeat(nrow, 1)
    #         scales = [75.0, 3.0, 112.0, 112.0]
    #         view_label[:, attr_idx] = attr_val / scales[attr_idx]
    #         return view_label # [nrow, 4]

    #     nrow = int(np.sqrt(args.batch_size))
    #     weight2 = torch.linspace(0.0, 1.0, nrow)
        
    #     # traversal_indicator = '110'
    #     # traversal_indicator = '101'
    #     traversal_indicator = '011'
        
    #     if traversal_indicator[0] == '1':
    #         ingr_label1 = create_ingr_label(1).repeat(nrow, 1)
    #         ingr_label2 = create_ingr_label(1).repeat(nrow, 1)
    #         ingr_label = torch.zeros(nrow, nrow, 10)
    #         for i in range(nrow):
    #             ingr_label[:, i] = ingr_label1 * (1-weight2[i]) + ingr_label2 * weight2[i]
    #         ingr_label = ingr_label.view(-1, 10)
    #     else:
    #         ingr_label = create_ingr_label(1).repeat(nrow*nrow, 1) 

    #     if traversal_indicator[1] == '1':
    #         attr_idx = Attr.ANGLE
    #         view_label1 = create_view_label(nrow, attr_idx, full_ranges[attr_idx][0])
    #         view_label2 = create_view_label(nrow, attr_idx, full_ranges[attr_idx][1])
    #         view_label = torch.zeros(nrow, nrow, 4)
    #         for i in range(nrow):
    #             view_label[:, i] = view_label1 * (1-weight2[i]) + view_label2 * weight2[i]     
    #         if traversal_indicator[0] == '0':
    #             view_label = view_label.view(-1, 4)
    #         else:
    #             view_label = view_label.transpose(0,1).contiguous().view(-1, 4)
    #     else:
    #         view_label = create_view_label(1, Attr.ANGLE, stats[Attr.ANGLE][0]*75.0).repeat(nrow*nrow, 1)

    #     label = torch.cat([ingr_label, view_label], dim=1)

    #     if traversal_indicator[2] == '1':
    #         z1 = torch.randn(1, 256).repeat(nrow, 1)
    #         z2 = torch.randn(1, 256).repeat(nrow, 1)
    #         z = torch.zeros(nrow, nrow, 256)
    #         for i in range(nrow):
    #             z[:, i] = z1 * (1-weight2[i]) + z2 * weight2[i]
    #         z = z.transpose(0,1).contiguous().view(-1, 256)
    #     else:
    #         z = torch.randn(1, 256).repeat(nrow*nrow, 1).to(device)

    #     with torch.no_grad():
    #         txt, fake = batch_generator.generate_from_label(label, z, randomize_noise=False)
        
    #     if traversal_indicator[1] == '1':
    #         fp = f'{save_dir}/{args.model_name}_trunc={args.truncation:.2f}_traversal={traversal_indicator}_attr={attr_idx}.png'
    #     else:
    #         fp = f'{save_dir}/{args.model_name}_trunc={args.truncation:.2f}_traversal={traversal_indicator}.png'

    #     common.save_captioned_image(txt, fake, fp, font=15, color=(255,255,0), opacity=0.2, loc=(0,0), nrow=int(np.sqrt(args.batch_size)), pad_value=1)
    #     print(f'saved to {fp}')