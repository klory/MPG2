import torch
import os
import numpy as np

import sys
sys.path.append('../')
from mpg.train import load_mpg
from datasets.pizza10 import Pizza10Dataset
from view_regressor.train import load_regressor
from common import infinite_loader, ROOT, requires_grad
from mpg.train import load_mpg
from datasets.pizza10 import Pizza10Dataset
from datasets.utils import gan_transform
from common import resize, normalize, requires_grad
from view_regressor.train import load_regressor

class BatchGenerator():
    def __init__(self, args):
        device = args.device
        ckpt_path = args.ckpt_path
        ckpt_args, _, label_encoder, _, _, netG, _, _, _ = load_mpg(ckpt_path, device=device)
        _, _, self.view_regressor, _  = load_regressor(args.view_regressor)
        self.view_regressor.eval()
        requires_grad(self.view_regressor, False)
        label_encoder = label_encoder.eval().to(device)
        netG = netG.eval().to(device)
        requires_grad(label_encoder, False)
        requires_grad(netG, False)
        if args.dataset == 'pizza10':
            dataset = Pizza10Dataset(transform=gan_transform)
        else:
            raise Exception('Unsupported dataset!')
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4)
        self.dataloader = infinite_loader(dataloader)
        self.ckpt_args = ckpt_args
        self.label_encoder = label_encoder
        self.netG = netG
        self.batch_size = args.batch_size
        self.truncation = args.truncation
        self.device = device
        self.fixed_z = torch.randn(args.batch_size, ckpt_args.style_dim, device=device)
        if args.truncation < 1:
            self.mean_latent = netG.mean_latent(args.truncation_mean)
        else:
            self.mean_latent = None

        self.args = args


    def generate_ssim(self):
        _, batch_img, batch_fake_img, _ = self.generate_all()
        return batch_img, batch_fake_img

    def generate_fid(self):
        _, _, batch_fake_img, _ = self.generate_all()
        return batch_fake_img

    def generate_MedR(self):
        batch_txt, _, batch_fake_img, _ = self.generate_all()
        return batch_txt, batch_fake_img

    def generate_mAP(self):
        _, _, batch_fake_img, batch_label = self.generate_all()
        return batch_fake_img, batch_label[:, :10]
    
    def generate_mAE(self):
        _, _, batch_fake_img, batch_label = self.generate_all()
        return batch_fake_img, batch_label[:, 10:]

    def generate_all(self, same_style=False, randomize_noise=True):
        real, target = next(self.dataloader)
        real = real.to(self.device)

        if 'pizza10' in self.args.dataset:
            ingr_label = target['ingr_label'].to(self.device)
            if ('random_view' in self.ckpt_args.__dict__ and self.ckpt_args.random_view) or ('random_label' in self.ckpt_args.__dict__ and self.ckpt_args.random_label):
                view_label = torch.rand(real.shape[0], 4).to(self.device)
                view_label[:,1] = view_label[:,1] * 0.67 + 0.33
                view_label[:,2] = (view_label[:,2]-0.5)/0.5
                view_label[:,3] = (view_label[:,3]-0.5)/0.5
            else:
                view_img = normalize(resize((real)))
                view_output = self.view_regressor(view_img)
                view_label = view_output
            label = torch.cat([ingr_label, view_label], dim=1)
            if same_style:
                z = self.fixed_z
            else:
                z = torch.randn_like(self.fixed_z)
            with torch.no_grad():
                txt_feat = self.label_encoder(label)
                fake, _ = self.netG([z], txt_feat, truncation=self.truncation, truncation_latent=self.mean_latent, randomize_noise=randomize_noise)
            return target['raw_label'], real, fake, label
            # txt: [BS], array of strings, e.g. [['Arugula\nTomato\nPepperoni'], ...]
            # real: [BS, 3, size, size]
            # fake: [BS, 3, size, size]
            # label: [BS, 14]
        elif 'celebAHQ' in self.args.dataset_name:
            label = target['label'].to(self.device)
            if same_style:
                z = self.fixed_z
            else:
                z = torch.randn_like(self.fixed_z)
            with torch.no_grad():
                txt_feat = self.label_encoder(label)
                fake, _ = self.netG(txt_feat, z, truncation=self.truncation, truncation_latent=self.mean_latent, randomize_noise=randomize_noise)
                batch_fake_img = fake
            return None, img, batch_fake_img, label


    # def generate_from_label(self, label, z, randomize_noise=True):
    #     assert len(label) == len(z)
    #     z = z.to(self.device)
    #     with torch.no_grad():
    #         txt_feat = self.label_encoder(label.to(self.device))
    #         if self.ckpt_args.encoder == 'simple*':
    #             z = torch.cat([txt_feat[:,0], z], dim=1)
    #             fake, _ = self.netG(z, randomize_noise=randomize_noise)
    #         else:
    #             fake, _ = self.netG(txt_feat, z, truncation=self.truncation, truncation_latent=self.mean_latent, randomize_noise=randomize_noise)

    #         batch_fake_img = fake

    #     batch_txt = []
    #     for one_label in label:
    #         one_label = one_label.cpu()
            
    #         one_label_ingr = one_label[:10]
    #         one_label_ingr_idxs = one_label_ingr.nonzero(as_tuple=False).view(-1)
    #         ingr_values = '\n'.join([f'{self.categories[i]} = {one_label_ingr[i]:.2f}' for i in one_label_ingr_idxs])
    #         one_txt = ingr_values
            
    #         if len(one_label)>10:
    #             one_label_view = (one_label[10:]*self.scales).tolist()
    #             view_attrs = ['angle', 'scale', 'dx', 'dy']
    #             view_values = '\n'.join([f'{view_attrs[i]} = {x:.2f}' for i,x in enumerate(one_label_view)])
    #             one_txt += '\n\n'
    #             one_txt += view_values
            
    #         batch_txt.append(one_txt)

    #     return batch_txt, batch_fake_img


if __name__ == '__main__':
    import pdb

    from types import SimpleNamespace
    args = SimpleNamespace(
        ckpt_path=f'{ROOT}/mpg_plusView/runs/1ehwwpy0/00220000.ckpt',
        batch_size=32,
        size=256,
        device='cuda',
        truncation=1.0,
        truncation_mean=4096
    )


    # pizzaGANdata_new_concise
    batch_generator = BatchGenerator(args)

    txt, img, fake_img, binary_label = batch_generator.generate_all()
    print(txt)
    print(img.shape)
    print(fake_img.shape)
    print(binary_label.shape)