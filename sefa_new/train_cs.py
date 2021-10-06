import argparse
import math
import random
import os
from types import GeneratorType

import numpy as np
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torchvision
from tqdm import tqdm
import pdb
import wandb

import sys
sys.path.append("../")
from common import ROOT
from datasets.pizza10 import Pizza10Dataset
from sefa_new.models import SimpleLabelEncoder, LabelEncoder, Generator, Discriminator
from mpg.non_leaking import augment
from torchvision import transforms
from common import infinite_loader, count_parameters
from ingr_classifier.train import load_classifier
# from angle_classifier.val_all import load_classifier as load_viewpoint_classifier

train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])

def data_sampler(dataset, shuffle):
    if shuffle:
        return data.RandomSampler(dataset)
    else:
        return data.SequentialSampler(dataset)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred, wrong_pred=None):
    """This loss works for both unconditonal and conditonal loss

    Args:
        real_pred (torch.tensor): D outputs for real images
        fake_pred (torch.tensor): D outputs for fake images
        wrong_pred (torch.tensor, optional): D outputs for contional wrong images. Defaults to None.

    Returns:
        torch.tensor: a scalar that averages all D outputs
    """
    real_loss = F.softplus(-real_pred) # real_pred -> max
    fake_loss = F.softplus(fake_pred) # fake_pred -> min

    if wrong_pred is None:
        return real_loss.mean() + fake_loss.mean()
    else:
        wrong_loss = F.softplus(wrong_pred) # wrong_pred -> min
        return real_loss.mean() + fake_loss.mean() + wrong_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


def g_path_regularize(fake_img, latents, mean_path_length, decay=0.01):
    noise = torch.randn_like(fake_img) / math.sqrt(
        fake_img.shape[2] * fake_img.shape[3]
    )
    pdb.set_trace()
    grad, = autograd.grad(
        outputs=(fake_img * noise).sum(), inputs=latents, create_graph=True,
    )
    path_lengths = torch.sqrt(grad.pow(2).sum(2).mean(1))

    path_mean = mean_path_length + decay * (path_lengths.mean() - mean_path_length)

    path_penalty = (path_lengths - path_mean).pow(2).mean()

    return path_penalty, path_mean.detach(), path_lengths


def make_noise(batch, latent_dim, n_noise, device):
    if n_noise == 1:
        return torch.randn(batch, latent_dim, device=device)

    noises = torch.randn(n_noise, batch, latent_dim, device=device).unbind(0)

    return noises


def mixing_noise(batch, latent_dim, prob, device):
    if prob > 0 and random.random() < prob:
        return make_noise(batch, latent_dim, 2, device)

    else:
        return [make_noise(batch, latent_dim, 1, device)]


def set_grad_none(model, targets):
    for n, p in model.named_parameters():
        if n in targets:
            p.grad = None



def load_mpg(ckpt_path):
    print(f'load mpg from {ckpt_path}')
    ckpt = torch.load(ckpt_path)
    ckpt_args = ckpt['args']


    label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim = create_mpg(ckpt_args)
    
    label_encoder.load_state_dict(ckpt['label_encoder'])
    generator.load_state_dict(ckpt["g"])
    discriminator.load_state_dict(ckpt["d"])
    g_ema.load_state_dict(ckpt["g_ema"])
    label_encoder_optim.load_state_dict(ckpt["label_encoder_optim"])
    g_optim.load_state_dict(ckpt["g_optim"])
    d_optim.load_state_dict(ckpt["d_optim"])

    return ckpt_args, 1, label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim


def create_mpg(args):
    args.z_dim=256
    device = args.device
    if 'encoder' not in args.__dict__ or args.encoder=='normal':
        label_encoder = LabelEncoder(
            size=args.size, input_dim=10, embed_dim=args.embed_dim
        ).to(device)
    elif args.encoder=='simple':
        label_encoder = SimpleLabelEncoder(
            size=args.size, input_dim=10, embed_dim=args.embed_dim
        ).to(device)
    generator = Generator(
        size=args.size, embed_dim=args.embed_dim, z_dim=args.z_dim, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, z_dim=args.z_dim
    ).to(device)
    g_ema = Generator(
       size=args.size, embed_dim=args.embed_dim, z_dim=args.z_dim, n_mlp=args.n_mlp, channel_multiplier=args.channel_multiplier
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    print(f'label_encoder parameters: {count_parameters(label_encoder)}')
    print(f'generator parameters: {count_parameters(generator)}')
    print(f'discriminator parameters: {count_parameters(discriminator)}')
    label_encoder_optim = optim.Adam(label_encoder.parameters(), lr=args.lr, betas=(0, 0.99))
    g_optim = optim.Adam(generator.parameters(), lr=args.lr, betas=(0, 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0, 0.99))
    return label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim
def handle_viewpoint_img_input(input_img):
    img = (input_img-input_img.min())/(input_img.max()-input_img.min())
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for channel in range(3):
        img[:,channel] = (img[:,channel]-means[channel])/stds[channel]
    # img: resize
    img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
    return img

def train_cond(args, loader, label_encoder, generator, discriminator, label_encoder_optim, g_optim, d_optim, g_ema, classifier,viewpoint_classifier, device, save_dir):
    img_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(img_save_dir, exist_ok=True)
    loader = infinite_loader(loader)

    pbar = range(args.iter)

    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.3)

    mean_path_length = 0

    r1_loss_cond = torch.tensor(0.0, device=device)
    r1_loss_uncond = torch.tensor(0.0, device=device)
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.device == 'cuda':
        label_encoder_module = label_encoder.module
        g_module = generator.module
        d_module = discriminator.module

    else:
        label_encoder_module = label_encoder
        g_module = generator
        d_module = discriminator


    accum = 0.5 ** (32 / (10 * 1000))
    ada_augment = torch.tensor([0.0, 0.0], device=device)
    ada_aug_p = args.augment_p if args.augment_p > 0 else 0.0 # 0.0
    ada_aug_step = args.ada_target / args.ada_length # 0.6 / 500k
    r_t_stat = 0

    sample_z = torch.randn(args.batch, args.style_dim, device=device)

    # save to file
    keys = [
        "Real Score Uncond", "Fake Score Uncond",
        "Real Score Cond", "Fake Score Cond", "Wrong Score Cond", 
        "D Loss Uncond", "D Loss Cond", "D Loss",
        "G Loss FID", "G Loss Cls", "G Loss Uncond", "G Loss Cond", "G Loss", 
        "Augment", "Rt", "R1 Loss Uncond", "R1 Loss Cond",
        "Path Length Regularization", "Mean Path Length", "Path Length", 
    ]
    f = open(os.path.join(save_dir, 'log.csv'), 'w')
    f.write(','.join(keys))
    f.write('\n')

    sample_txt, sample_img, _, sample_binary_label = next(loader)
    sample_viewpoint_img = handle_viewpoint_img_input(sample_img)
    sample_viewpoint = viewpoint_classifier(sample_viewpoint_img.to(device))
    sample_viewpoint_label = sample_viewpoint[:,0:args.viewpoint_dim]
    with open(os.path.join(save_dir, f'real_txt.txt'), 'w') as file:
        for i,txt in enumerate(sample_txt):
            file.write(str(i+1)+'\n')
            file.write(txt)
            file.write('\n')
    torchvision.utils.save_image(
        sample_img,
        os.path.join(save_dir, f"real_img.png"),
        nrow=int(args.batch ** 0.5),
        normalize=True,
        range=(-1, 1),
    )

    # classifier regularier
    cls_criterion = nn.BCEWithLogitsLoss()
    # regression regularier
    reg_criterion = nn.MSELoss()
    # FIDLoss
    if 'fid' not in args.__dict__:
        args.fid = 0.0
    if args.fid:
        sys.path.append('../fid_loss')
        from fid_loss import FIDLoss
        import pickle
        if 'pizzaGANdata_new_concise' in args.lmdb_file:
            with open('../metrics/pizzaGANdata_new_concise.pkl', 'rb') as f_stat:
                embeds = pickle.load(f_stat)
                real_mean = torch.tensor(embeds['mean']).to(device)
                real_cov = torch.tensor(embeds['cov']).to(device)
        fid_criterion = FIDLoss(real_mean, real_cov)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")
            break

        _, real_img, wrong_img, binary_label = next(loader)
        real_img = real_img.to(device)
        wrong_img = wrong_img.to(device)
        binary_label = binary_label.to(device)
        viewpoint_img = handle_viewpoint_img_input(real_img)
        viewpoint_output = viewpoint_classifier(viewpoint_img)
        viewpoint_label = viewpoint_output[:,0:args.viewpoint_dim]

        requires_grad(label_encoder, False)
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        noise = mixing_noise(args.batch, args.style_dim, args.mixing, device)

        text_outputs = label_encoder(torch.cat((binary_label,viewpoint_label),dim=1))
        fake_img, _ = generator(noise, text_outputs, input_is_latent=args.input_is_latent)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)

        else:
            real_img_aug = real_img
        
        if args.uncond>0:
            # pdb.set_trace()
            fake_pred_uncond = discriminator(fake_img)
            real_pred_uncond = discriminator(real_img_aug)
            d_loss_uncond = d_logistic_loss(real_pred_uncond, fake_pred_uncond)
        else:
            fake_pred_uncond = real_pred_uncond = d_loss_uncond = torch.tensor(0.0, device=device)
    
        fake_pred_cond = discriminator(fake_img, text_outputs)
        real_pred_cond = discriminator(real_img, text_outputs)
        wrong_pred_cond = discriminator(wrong_img, text_outputs)
        if args.wrong:
            d_loss_cond = d_logistic_loss(real_pred_cond, fake_pred_cond, wrong_pred_cond)
        else:
            d_loss_cond = d_logistic_loss(real_pred_cond, fake_pred_cond, None)

        d_loss = args.uncond * d_loss_uncond + args.cond * d_loss_cond

        loss_dict["real_score_uncond"] = real_pred_uncond.mean()
        loss_dict["fake_score_uncond"] = fake_pred_uncond.mean()
        loss_dict["real_score_cond"] = real_pred_cond.mean()
        loss_dict["fake_score_cond"] = fake_pred_cond.mean()
        loss_dict["wrong_score_cond"] = wrong_pred_cond.mean()
        loss_dict["d_loss_uncond"] = d_loss_uncond
        loss_dict["d_loss_cond"] = d_loss_cond
        loss_dict["d_loss"] = d_loss

        discriminator.zero_grad()
        d_loss.backward()
        d_optim.step()

        if args.augment and args.augment_p == 0:
            if args.uncond > 0:
                ada_augment_data_uncond = torch.tensor(
                    (torch.sign(real_pred_uncond).sum().item(), real_pred_uncond.shape[0]), device=device
                )
                ada_augment += ada_augment_data_uncond

            ada_augment_data_cond = torch.tensor(
                (torch.sign(real_pred_cond).sum().item(), real_pred_cond.shape[0]), device=device
            )
            ada_augment += ada_augment_data_cond

            if ada_augment[1] > 255:
                pred_signs, n_pred = ada_augment.tolist()

                r_t_stat = pred_signs / n_pred

                if r_t_stat > args.ada_target: # => overfitted
                    sign = 1

                else: # not overfit
                    sign = -1

                ada_aug_p += sign * ada_aug_step * n_pred # each image will increase/decrease ∆p=ada_aug_step
                ada_aug_p = min(1, max(0, ada_aug_p))
                ada_augment.mul_(0)

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True
            real_pred_cond = discriminator(real_img, text_outputs)
            r1_loss_cond = d_r1_loss(real_pred_cond, real_img)

            if args.uncond > 0:
                real_pred_uncond = discriminator(real_img)
                r1_loss_uncond = d_r1_loss(real_pred_uncond, real_img)
            else:
                r1_loss_uncond = torch.tensor(0.0, device=device)

            discriminator.zero_grad()
            (args.r1 / 2 * (r1_loss_cond+r1_loss_uncond) * args.d_reg_every + 0 * real_pred_cond[0]).backward()

            d_optim.step()

        loss_dict["r1_loss_cond"] = r1_loss_cond
        loss_dict["r1_loss_uncond"] = r1_loss_uncond

        requires_grad(label_encoder, True)
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        noise = mixing_noise(args.batch, args.style_dim, args.mixing, device)
        
        text_outputs = label_encoder(torch.cat((binary_label,viewpoint_label),dim=1))
        fake_img, _ = generator(noise, text_outputs, input_is_latent=args.input_is_latent)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        g_loss_fid = torch.tensor(0.0)
        if args.fid:
            g_loss_fid = fid_criterion(fake_img)

        g_loss_cls = torch.tensor(0.0)
        g_loss_viewpoint = torch.tensor(0.0)
        if args.cls:
            # img: normalize
            img = (fake_img-fake_img.min())/(fake_img.max()-fake_img.min())
            means = [0.485, 0.456, 0.406]
            stds = [0.229, 0.224, 0.225]
            for channel in range(3):
                img[:,channel] = (img[:,channel]-means[channel])/stds[channel]
            # img: resize
            img = F.interpolate(img, size=(224, 224), mode='bilinear', align_corners=False)
            # retrieve
            output = classifier(img)

            g_loss_cls = cls_criterion(output, binary_label.float())
            output = viewpoint_classifier(img)
            # print(output[:,0:args.viewpoint_dim])
            # print(viewpoint_label)
            g_loss_viewpoint= reg_criterion(output[:,0:args.viewpoint_dim], viewpoint_label.float())
            
        if args.uncond > 0:
            fake_pred_uncond = discriminator(fake_img)
            g_loss_uncond = g_nonsaturating_loss(fake_pred_uncond)
        else:
            g_loss_uncond = torch.tensor(0.0, device=device)

        fake_pred_cond = discriminator(fake_img, text_outputs)
        g_loss_cond = g_nonsaturating_loss(fake_pred_cond)

        g_loss = args.uncond * g_loss_uncond + args.cond * g_loss_cond + args.cls * g_loss_cls + args.fid * g_loss_fid+args.viewpoint*g_loss_viewpoint
        
        loss_dict["g_loss_fid"] = g_loss_fid
        loss_dict["g_loss_cls"] = g_loss_cls
        loss_dict["g_loss_uncond"] = g_loss_uncond
        loss_dict["g_loss_cond"] = g_loss_cond
        loss_dict["g_loss"] = g_loss
        loss_dict["g_loss_view"] = g_loss_viewpoint
        label_encoder.zero_grad()
        generator.zero_grad()
        g_loss.backward()
        label_encoder_optim.step()
        g_optim.step()

        # *************************
        # NOTE: generator regularizer is not working on multi-gpu training for now
        # g_regularize = i % args.g_reg_every == 0
        g_regularize = 0
        # *************************

        if g_regularize:
            path_batch_size = max(1, args.batch // args.path_batch_shrink)
            noise = mixing_noise(path_batch_size, args.style_dim, args.mixing, device)
            text_outputs = label_encoder(binary_label)
            fake_img, latents = generator(noise, text_outputs, input_is_latent=args.input_is_latent, return_latents=True)

            path_loss, mean_path_length, path_lengths = g_path_regularize(
                fake_img, latents, mean_path_length
            )

            label_encoder.zero_grad()
            generator.zero_grad()
            weighted_path_loss = args.path_regularize * args.g_reg_every * path_loss

            if args.path_batch_shrink:
                weighted_path_loss += 0 * fake_img[0, 0, 0, 0]

            weighted_path_loss.backward()
            
            label_encoder_optim.step()
            g_optim.step()

            # mean_path_length_avg = (
            #     reduce_sum(mean_path_length).item() / get_world_size()
            # )
            mean_path_length_avg = mean_path_length

        loss_dict["path_loss"] = path_loss
        loss_dict["path_length"] = path_lengths.mean()

        accumulate(g_ema, g_module, accum)

        loss_reduced = loss_dict 

        real_score_uncond_val = loss_reduced["real_score_uncond"].mean().item()
        fake_score_uncond_val = loss_reduced["fake_score_uncond"].mean().item()
        
        real_score_cond_val = loss_reduced["real_score_cond"].mean().item()
        fake_score_cond_val = loss_reduced["fake_score_cond"].mean().item()
        wrong_score_cond_val = loss_reduced["wrong_score_cond"].mean().item()
        
        d_loss_uncond_val = loss_reduced["d_loss_uncond"].mean().item()
        d_loss_cond_val = loss_reduced["d_loss_cond"].mean().item()
        d_loss_val = loss_reduced["d_loss"].mean().item()
        
        g_loss_fid_val = loss_reduced["g_loss_fid"].mean().item()
        g_loss_cls_val = loss_reduced["g_loss_cls"].mean().item()
        g_loss_uncond_val = loss_reduced["g_loss_uncond"].mean().item()
        g_loss_cond_val = loss_reduced["g_loss_cond"].mean().item()
        g_loss_val = loss_reduced["g_loss"].mean().item()
        g_loss_view_val = loss_reduced["g_loss_view"].mean().item()
        
        r1_loss_uncond_val = loss_reduced["r1_loss_uncond"].mean().item()
        r1_loss_cond_val = loss_reduced["r1_loss_cond"].mean().item()
        
        path_loss_val = loss_reduced["path_loss"].mean().item()
        path_length_val = loss_reduced["path_length"].mean().item()

        pbar.set_description(
            (
                f"d: {d_loss_val:.4f}; g: {g_loss_val:.4f}; r1 loss cond: {r1_loss_cond_val:.4f}; "
                f"path loss: {path_loss_val:.4f}; mean path: {mean_path_length_avg:.4f}; "
                f"augment: {ada_aug_p:.4f}"
            )
        )

        log = {
            "Real Score Uncond": real_score_uncond_val,
            "Fake Score Uncond": fake_score_uncond_val,

            "Real Score Cond": real_score_cond_val,
            "Fake Score Cond": fake_score_cond_val,
            "Wrong Score Cond": wrong_score_cond_val,

            "D Loss Uncond": d_loss_uncond_val,
            "D Loss Cond": d_loss_cond_val,
            "D Loss": d_loss_val,

            "G Loss FID": g_loss_fid_val,
            "G Loss Cls": g_loss_cls_val,
            "G Loss view": g_loss_view_val,
            "G Loss Uncond": g_loss_uncond_val,
            "G Loss Cond": g_loss_cond_val,
            "G Loss": g_loss_val,

            "Augment": ada_aug_p,
            "Rt": r_t_stat,
            "R1 Loss Uncond": r1_loss_uncond_val,
            "R1 Loss Cond": r1_loss_cond_val,

            "Path Length Regularization": path_loss_val,
            # "Mean Path Length": mean_path_length.item(),
            "Mean Path Length": mean_path_length,
            "Path Length": path_length_val,
        }
        line = ','.join([str(log[k]) for k in keys])
        f.write(line)
        f.write('\n')

        wandb.log(log)

        if i % 100 == 0:
            with torch.no_grad():
                g_ema.eval()
                text_outputs = label_encoder(torch.cat((sample_binary_label,sample_viewpoint_label.to('cpu')),dim=1))
                sample, _ = g_ema([sample_z], text_outputs, input_is_latent=args.input_is_latent)
                torchvision.utils.save_image(
                    sample,
                    os.path.join(img_save_dir, f"{str(i).zfill(6)}.png"),
                    nrow=int(args.batch ** 0.5),
                    normalize=True,
                    range=(-1, 1),
                )

        if i % 10000 == 0:
            filename = os.path.join(save_dir, f"{str(i).zfill(6)}.pt")
            print(f'saving mpg to {filename}')
            torch.save(
                {
                    'label_encoder': label_encoder_module.state_dict(),
                    "g": g_module.state_dict(),
                    "d": d_module.state_dict(),
                    "g_ema": g_ema.state_dict(),
                    'label_encoder_optim': label_encoder_optim.state_dict(),
                    "g_optim": g_optim.state_dict(),
                    "d_optim": d_optim.state_dict(),
                    "ada_aug_p": ada_aug_p,
                    "args": args,
                },
                filename,
            )
    f.close()


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lmdb_file", type=str, 
        default=f'{ROOT}/data/pizzaGANdata_new_concise/pizzaGANdata.lmdb')
    parser.add_argument("--classifier_path", type=str, default='../ingr_classifier/runs/1t5xrvwx/batch5000.ckpt')
    # parser.add_argument("--viewpoint_classifier_path", type=str, default='../angle_classifier/runs/2lfc8wxr/batch2000.ckpt')
    # parser.add_argument("--ckpt_path", type=str, default='../mpg_plusView/runs/2e1wjaiz/020000.pt')
    parser.add_argument("--viewpoint_classifier_path", type=str, default='../angle_classifier/runs/dodui0og/batch4999.ckpt')
    parser.add_argument("--ckpt_path", type=str, default='')
    parser.add_argument("--encoder", type=str, default='normal', choices=['simple', 'normal'])

    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--input_is_latent", type=int, default=0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--style_dim", type=int, default=256)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--path_regularize", type=float, default=2)
    # parser.add_argument("--path_batch_shrink", type=int, default=2)
    parser.add_argument("--path_batch_shrink", type=int, default=1, choices=[1])
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)
    parser.add_argument("--mixing", type=float, default=0.9)

    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--wrong", type=int, default=1, choices=[0,1])
    parser.add_argument("--cond", type=float, default=1.0)
    parser.add_argument("--uncond", type=float, default=1.0)
    parser.add_argument("--cls", type=float, default=1.0)
    parser.add_argument("--fid", type=float, default=0.0)

    parser.add_argument("--viewpoint", type=float, default=1.0)
    parser.add_argument("--viewpoint_dim", type=int, default=4)

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    torch.backends.cudnn.benchmark = True

    if args.ckpt_path:
        ckpt_args, batch, label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim = load_mpg(args.ckpt_path, device=device)
        args.start_iter = batch + 1
    else:
        args.start_iter = 0
        label_encoder, generator, discriminator, g_ema = create_mpg(args, device)

        g_reg_ratio = args.g_reg_every / (args.g_reg_every + 1)
        d_reg_ratio = args.d_reg_every / (args.d_reg_every + 1)

        label_encoder_optim = optim.Adam(
            label_encoder.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        g_optim = optim.Adam(
            generator.parameters(),
            lr=args.lr * g_reg_ratio,
            betas=(0 ** g_reg_ratio, 0.99 ** g_reg_ratio),
        )
        d_optim = optim.Adam(
            discriminator.parameters(),
            lr=args.lr * d_reg_ratio,
            betas=(0 ** d_reg_ratio, 0.99 ** d_reg_ratio),
        )
    
    if args.device == 'cuda':
        label_encoder, generator, discriminator = [nn.DataParallel(x) for x in [label_encoder, generator, discriminator]]

    _, _, classifier, _ = load_classifier(args.classifier_path)
    _,_,viewpoint_classifier = load_viewpoint_classifier(args.viewpoint_classifier_path)
    viewpoint_classifier = viewpoint_classifier.eval().to(device)
    requires_grad(viewpoint_classifier, False)
    classifier = classifier.eval().to(device)
    requires_grad(classifier, False)
    dataset = Pizza10Dataset(
        lmdb_file=args.lmdb_file, 
        transform=train_transform, resolution=args.size)
    
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        num_workers=args.workers,
        sampler=data_sampler(dataset, shuffle=True),
        drop_last=True,
    )

    wandb.init(project="mpg")
    wandb.config.update(args)

    save_dir = os.path.join(os.path.dirname(__file__), 'runs', wandb.run.id)
    os.makedirs(save_dir, exist_ok=True)
    print(f'save_dir: {save_dir}')

    train_cond(args, loader, label_encoder, generator, discriminator, label_encoder_optim, g_optim, d_optim, g_ema, classifier,viewpoint_classifier, device, save_dir)