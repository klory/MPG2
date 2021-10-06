import argparse
import os
import torch
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
from tqdm import tqdm
import pdb
import scipy

import sys
sys.path.append("../")
from common import infinite_loader, count_parameters, requires_grad, save_captioned_image
from datasets.pizza10 import Pizza10Dataset
from datasets.pizza3d import denormalize_view_label
from datasets.utils import gan_transform, gan_transform_aug
from mpg.models import OneLabelEncoder, LabelEncoder, Generator, Discriminator
from mpg.non_leaking import augment
from ingr_classifier.train import load_classifier as load_ingr_classifier
from view_regressor.train import load_regressor as load_view_regressor
from common import normalize, resize

def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def d_logistic_loss(real_pred, fake_pred, weight=None):
    """This loss works for both unconditonal and conditonal loss

    Args:
        real_pred (torch.tensor): D outputs for real images
        fake_pred (torch.tensor): D outputs for fake images

    Returns:
        torch.tensor: a scalar that averages all D outputs
    """
    real_loss = F.softplus(-real_pred) # real_pred -> max
    fake_loss = F.softplus(fake_pred) # fake_pred -> min
    if weight is not None:
        real_loss = weight * real_loss
        fake_loss = weight * fake_loss
    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    grad_real, = autograd.grad(
        outputs=real_pred.sum(), inputs=real_img, create_graph=True
    )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred, weight=None):
    fake_loss = F.softplus(-fake_pred)
    if weight is not None:
        fake_loss = weight * fake_loss
    return fake_loss.mean()


def load_mpg(ckpt_path, device='cuda'):
    print(f'load mpg from {ckpt_path}')
    ckpt_name = os.path.basename(ckpt_path)
    batch = int(os.path.splitext(ckpt_name)[0])

    ckpt = torch.load(ckpt_path)
    ckpt_args = ckpt['args']
    label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim = create_mpg(ckpt_args, device)
    label_encoder_optim = optim.Adam(label_encoder.parameters())
    g_optim = optim.Adam(generator.parameters())
    d_optim = optim.Adam(discriminator.parameters())

    label_encoder.load_state_dict(ckpt['label_encoder'])
    generator.load_state_dict(ckpt["g"])
    discriminator.load_state_dict(ckpt["d"])
    g_ema.load_state_dict(ckpt["g_ema"])

    label_encoder_optim.load_state_dict(ckpt["label_encoder_optim"])
    g_optim.load_state_dict(ckpt["g_optim"])
    d_optim.load_state_dict(ckpt["d_optim"])

    return ckpt_args, batch, label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim

def create_mpg(args, device='cuda'):
    if args.encoder == 'cat_z':
        label_encoder = OneLabelEncoder(input_dim=args.num_attributes, embed_dim=args.embed_dim, n_layers=args.n_encoder_layers).to(device)
    else:
        label_encoder = LabelEncoder(
            size=args.size, input_dim=args.num_attributes, 
            embed_dim=args.embed_dim, n_layers=args.n_encoder_layers, type_=args.encoder).to(device)
    
    generator = Generator(
        size=args.size, embed_dim=args.embed_dim, 
        style_dim=args.style_dim, n_mlp=args.n_mlp, 
        channel_multiplier=args.channel_multiplier, cat_z=args.encoder=='cat_z'
    ).to(device)
    discriminator = Discriminator(
        args.size, channel_multiplier=args.channel_multiplier, style_dim=args.style_dim
    ).to(device)
    g_ema = Generator(
       size=args.size, embed_dim=args.embed_dim, 
       style_dim=args.style_dim, n_mlp=args.n_mlp, 
       channel_multiplier=args.channel_multiplier, cat_z=args.encoder=='cat_z'
    ).to(device)
    g_ema.eval()
    accumulate(g_ema, generator, 0)

    print(f'label_encoder parameters: {count_parameters(label_encoder)}')
    print(f'generator parameters: {count_parameters(generator)}')
    print(f'discriminator parameters: {count_parameters(discriminator)}')

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

    return label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim


def estimate_density_for_view_attributes(view_attributes, stats):
    device = view_attributes.device
    density = []
    for i in range(4):
        w = scipy.stats.distributions.norm.pdf(view_attributes[:, i].cpu(), stats[i][0], stats[i][1])
        w = torch.FloatTensor(w).to(device)
        density.append(w)
    density = torch.stack(density, dim=0).t() # [BS, 4]
    return density + 1e-8


def compute_weight_for_gan_loss(density_each_attr):
    weight_each_sample = torch.prod(density_each_attr, dim=1)
    # normalize
    weight_each_sample = 1.0 / weight_each_sample
    weight_each_sample = weight_each_sample / weight_each_sample.sum() * len(weight_each_sample)
    return weight_each_sample


def weighted_mse_loss(input, target, weight):
    return torch.sum(weight * (input - target) ** 2)


def snapshot(sample_imgs, sample_ingrs, sample_views, img_save_dir, img_name='sample_imgs.jpg'):
    caption = []
    for idx in range(len(sample_ingrs)):
        cap = sample_ingrs[idx] + '\n'
        view = '|'.join([f'{x:.1f}' for x in sample_views[idx]])
        cap += view
        caption.append(cap)
    save_captioned_image(caption, sample_imgs, os.path.join(img_save_dir, img_name))



def train_cond(args, loader, label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim, ingr_classifier, view_regressor):
    device = args.device
    save_dir = args.save_dir
    img_save_dir = os.path.join(save_dir, 'images')
    os.makedirs(img_save_dir, exist_ok=True)
    loader = infinite_loader(loader)

    pbar = range(args.iter)
    pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True, smoothing=0.3)

    r1_loss_cond = torch.tensor(0.0, device=device)
    r1_loss_uncond = torch.tensor(0.0, device=device)
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

    # fixed_z for validation
    sample_z = torch.randn(args.batch_size, args.style_dim).to(device)
    sample_img, sample_tgt = next(loader)
    sample_view_img = normalize(resize((sample_img)))
    sample_view_label = view_regressor(sample_view_img.to(device))
    sample_views = denormalize_view_label(sample_view_label)
    sample_ingrs = sample_tgt['raw_label']
    sample_ingr_label = sample_tgt['ingr_label'].to(device)

    snapshot(sample_img, sample_ingrs, sample_views, img_save_dir=img_save_dir, img_name='sample_img.jpg')
    # ingr_classifier regularier
    cls_criterion = nn.BCEWithLogitsLoss()
    # regression regularier
    reg_criterion = nn.MSELoss()

    for idx in pbar:
        batch_idx = idx + args.start_iter

        if batch_idx > args.iter:
            print("Done!")
            break

        real_img, tgt = next(loader)
        ingr_label = tgt['ingr_label']
        real_img = real_img.to(device)
        ingr_label = ingr_label.to(device)
        view_img = normalize(resize((real_img)))
        view_output = view_regressor(view_img)
        view_label = view_output

        # ***********************
        # optimize discriminator
        # ***********************
        requires_grad(label_encoder, False)
        requires_grad(generator, False)
        requires_grad(discriminator, True)

        weight = None
        real_label_embedding = label_encoder(torch.cat((ingr_label, view_label), dim=1))
        label_embedding = label_encoder(torch.cat((ingr_label, view_label), dim=1))
        z = torch.randn(args.batch_size, args.style_dim, device=device)
        # print(z.shape, label_embedding.shape)
        fake_img, _ = generator([z], label_embedding, input_is_latent=args.input_is_latent)

        if args.augment:
            real_img_aug, _ = augment(real_img, ada_aug_p)
            fake_img, _ = augment(fake_img, ada_aug_p)
        else:
            real_img_aug = real_img
        
        if args.uncond>0:
            # pdb.set_trace()
            fake_pred_uncond = discriminator(fake_img)
            real_pred_uncond = discriminator(real_img_aug)
            d_loss_uncond = d_logistic_loss(real_pred_uncond, fake_pred_uncond, weight=weight)
        else:
            fake_pred_uncond = real_pred_uncond = d_loss_uncond = torch.tensor(0.0).to(device)

        if args.cond>0:
            fake_pred_cond = discriminator(fake_img, label_embedding)
            real_pred_cond = discriminator(real_img, real_label_embedding)
            d_loss_cond = d_logistic_loss(real_pred_cond, fake_pred_cond, weight=weight)
        else:
            fake_pred_cond = real_pred_cond = d_loss_cond = torch.tensor(0.0).to(device)

        d_loss = args.uncond * d_loss_uncond + args.cond * d_loss_cond

        loss_dict["real_score_uncond"] = real_pred_uncond.mean()
        loss_dict["fake_score_uncond"] = fake_pred_uncond.mean()
        loss_dict["d_loss_uncond"] = d_loss_uncond
        loss_dict["real_score_cond"] = real_pred_cond.mean()
        loss_dict["fake_score_cond"] = fake_pred_cond.mean()
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

        d_regularize = batch_idx % args.d_reg_every == 0

        if d_regularize:
            real_img.requires_grad = True

            if args.uncond > 0:
                real_pred_uncond = discriminator(real_img)
                r1_loss_uncond = d_r1_loss(real_pred_uncond, real_img)
            else:
                r1_loss_uncond = torch.tensor(0.0, device=device)
            
            if args.cond > 0:
                real_pred_cond = discriminator(real_img, real_label_embedding)
                r1_loss_cond = d_r1_loss(real_pred_cond, real_img)
            else:
                r1_loss_cond = torch.tensor(0.0, device=device)

            discriminator.zero_grad()
            (args.r1 * (r1_loss_cond+r1_loss_uncond) * args.d_reg_every).backward()
            d_optim.step()

        loss_dict["r1_loss_cond"] = r1_loss_cond
        loss_dict["r1_loss_uncond"] = r1_loss_uncond

        # ***********************
        # optimize generator
        # ***********************
        requires_grad(label_encoder, True)
        requires_grad(generator, True)
        requires_grad(discriminator, False)

        z = torch.randn(args.batch_size, args.style_dim, device=device)
        label_embedding = label_encoder(torch.cat((ingr_label, view_label), dim=1))
        fake_img, _ = generator([z], label_embedding, input_is_latent=args.input_is_latent)

        if args.augment:
            fake_img, _ = augment(fake_img, ada_aug_p)

        g_loss_ingr = torch.tensor(0.0)
        if args.ingr_cls:
            fake_img_for_classifier = normalize(resize(fake_img))
            output = ingr_classifier(fake_img_for_classifier)
            g_loss_ingr = cls_criterion(output, ingr_label.float())

        g_loss_view = torch.tensor(0.0)
        if args.view_cls:
            output = view_regressor(fake_img_for_classifier)
            weight = torch.ones_like(view_label)
            g_loss_view = weighted_mse_loss(output, view_label, weight)
            
        if args.uncond > 0:
            fake_pred_uncond = discriminator(fake_img)
            g_loss_uncond = g_nonsaturating_loss(fake_pred_uncond, weight=weight)
        else:
            g_loss_uncond = torch.tensor(0.0).to(device)

        if args.cond > 0:
            fake_pred_cond = discriminator(fake_img, label_embedding)
            g_loss_cond = g_nonsaturating_loss(fake_pred_cond, weight=weight)
        else:
            g_loss_cond = torch.tensor(0.0).to(device)

        g_loss = args.uncond*g_loss_uncond + args.cond*g_loss_cond + args.ingr_cls*g_loss_ingr + args.view_cls*g_loss_view
        
        loss_dict["g_loss_uncond"] = g_loss_uncond
        loss_dict["g_loss_cond"] = g_loss_cond
        loss_dict["g_loss_ingr"] = g_loss_ingr
        loss_dict["g_loss_view"] = g_loss_view
        loss_dict["g_loss"] = g_loss

        label_encoder.zero_grad()
        generator.zero_grad()
        g_loss.backward()
        label_encoder_optim.step()
        g_optim.step()


        accumulate(g_ema, g_module, accum)

        loss_reduced = loss_dict 

        real_score_uncond_val = loss_reduced["real_score_uncond"].mean().item()
        fake_score_uncond_val = loss_reduced["fake_score_uncond"].mean().item()
        real_score_cond_val = loss_reduced["real_score_cond"].mean().item()
        fake_score_cond_val = loss_reduced["fake_score_cond"].mean().item()
        d_loss_uncond_val = loss_reduced["d_loss_uncond"].mean().item()
        d_loss_cond_val = loss_reduced["d_loss_cond"].mean().item()
        d_loss_val = loss_reduced["d_loss"].mean().item()
        
        g_loss_uncond_val = loss_reduced["g_loss_uncond"].mean().item()
        g_loss_cond_val = loss_reduced["g_loss_cond"].mean().item()
        g_loss_ingr_val = loss_reduced["g_loss_ingr"].mean().item()
        g_loss_view_val = loss_reduced["g_loss_view"].mean().item()
        g_loss_val = loss_reduced["g_loss"].mean().item()
        
        r1_loss_uncond_val = loss_reduced["r1_loss_uncond"].mean().item()
        r1_loss_cond_val = loss_reduced["r1_loss_cond"].mean().item()
        
        pbar.set_description(
            (
                f"d uncond: {d_loss_uncond_val:.4f}, d cond: {d_loss_cond_val:.4f}, d r1 uncond: {r1_loss_uncond_val:.4f}, d r1 cond: {r1_loss_cond_val:.4f}; "
                f"g uncond: {g_loss_uncond_val:.4f}, g cond: {g_loss_cond_val:.4f}, g ingr: {g_loss_ingr_val:.4f}, g view: {g_loss_view_val:.4f}; "
                f"augment: {ada_aug_p:.4f}"
            )
        )

        log = {
            "Real Score Uncond": real_score_uncond_val,
            "Fake Score Uncond": fake_score_uncond_val,
            "Real Score Cond": real_score_cond_val,
            "Fake Score Cond": fake_score_cond_val,

            "D Loss Uncond": d_loss_uncond_val,
            "D Loss Cond": d_loss_cond_val,
            "D Loss": d_loss_val,

            "G Loss Uncond": g_loss_uncond_val,
            "G Loss Cond": g_loss_cond_val,
            "G Loss ingr": g_loss_ingr_val,
            "G Loss view": g_loss_view_val,
            "G Loss": g_loss_val,

            "Augment": ada_aug_p,
            "Rt": r_t_stat,
            "R1 Loss Uncond": r1_loss_uncond_val,
            "R1 Loss Cond": r1_loss_cond_val,
        }

        if args.wandb:
            wandb.log(log)

        if batch_idx % 100 == 0:
            with torch.no_grad():
                g_ema.eval()
                sample_label_embedding = label_encoder(torch.cat([sample_ingr_label, sample_view_label],dim=1))
                sample_img, _ = generator([sample_z], sample_label_embedding, input_is_latent=args.input_is_latent)
                snapshot(sample_img, sample_ingrs, sample_views, img_save_dir=img_save_dir, img_name=f'{batch_idx:>08d}.jpg')

        if batch_idx % 10000 == 0:
            filename = os.path.join(save_dir, f"{batch_idx:>08d}.ckpt")
            print(f'saving model to {filename}')
            torch.save(
                {
                    'batch_idx': batch_idx,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default='pizza10', choices=['pizza10', 'pizza3d'])
    parser.add_argument("--ingr_classifier_path", type=str, default='../ingr_classifier/runs/pizza10/1t5xrvwx/batch5000.ckpt')
    parser.add_argument("--view_regressor_path", type=str, default='../view_regressor/runs/pizza3d/1ab8hru7/00004999.ckpt')
    parser.add_argument("--ckpt_path", type=str, default='')
    parser.add_argument("--device", type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument("--seed", type=int, default=8)

    parser.add_argument("--num_attributes", type=int, default=14)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=8)

    parser.add_argument("--transform", type=str, default='normal', choices=['augmented', 'normal'])
    parser.add_argument("--iter", type=int, default=800000)
    parser.add_argument("--n_sample", type=int, default=64)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--n_mlp", type=int, default=8)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--augment", action="store_true")
    parser.add_argument("--augment_p", type=float, default=0)
    parser.add_argument("--ada_target", type=float, default=0.6)
    parser.add_argument("--ada_length", type=int, default=500 * 1000)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--g_reg_every", type=int, default=4)

    parser.add_argument("--wandb", type=int, default=1, choices=[0,1])
    parser.add_argument("--input_is_latent", type=int, default=0, choices=[0,1])
    parser.add_argument("--encoder", type=str, default='many', choices=['one', 'many', 'cat_z'])
    parser.add_argument("--n_encoder_layers", type=int, default=1)
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--style_dim", type=int, default=256)
    parser.add_argument("--cond", type=float, default=0.0)
    parser.add_argument("--uncond", type=float, default=1.0)
    parser.add_argument("--ingr_cls", type=float, default=1.0)
    parser.add_argument("--view_cls", type=float, default=1.0)

    args = parser.parse_args()

    assert args.batch_size%2 == 0, 'batch_size has to be an even number'

    torch.manual_seed(args.seed)
    device = args.device
    torch.backends.cudnn.benchmark = True

    if args.ckpt_path:
        ckpt_args, batch_idx, label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim = load_mpg(args.ckpt_path)
        args.start_iter = batch_idx + 1
    else:
        label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim = create_mpg(args)
        args.start_iter = 0
    
    if args.device == 'cuda':
        label_encoder, generator, discriminator = [nn.DataParallel(x) for x in [label_encoder, generator, discriminator]]

    ingr_classifier = None
    view_regressor = None
    if args.dataset == 'pizza10':
        _, _, ingr_classifier, _ = load_ingr_classifier(args.ingr_classifier_path)
        requires_grad(ingr_classifier, False)
        ingr_classifier = ingr_classifier.eval().to(device)
        _, _, view_regressor, _ = load_view_regressor(args.view_regressor_path)
        view_regressor = view_regressor.eval().to(device)
        requires_grad(view_regressor, False)
    
    if args.transform == 'normal':
        dataset = Pizza10Dataset(part='all', transform=gan_transform)
    else:
        dataset = Pizza10Dataset(part='all', transform=gan_transform_aug)
    
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        drop_last=True,
    )

    if args.wandb:
        import wandb
        wandb.init(project="MPG_BMVC21_mpg")
        wandb.config.update(args)
        save_dir = os.path.join(os.path.dirname(__file__), 'runs', wandb.run.id)
    else:
        from datetime import datetime
        dateTimeObj = datetime.now()
        time_stamp = dateTimeObj.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join(os.path.dirname(__file__), 'runs', time_stamp)
    
    os.makedirs(save_dir, exist_ok=True)
    print(f'save_dir: {save_dir}')
    args.save_dir = save_dir

    train_cond(args, loader, label_encoder, generator, discriminator, g_ema, label_encoder_optim, g_optim, d_optim, ingr_classifier, view_regressor)
    
    wandb.join()