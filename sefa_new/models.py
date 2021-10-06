import math
from os import wait3
import random

import torch
from torch import nn
from torch.nn import functional as F

import sys
sys.path.append('../')
from mpg.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

def parse_gan_type(generator):
    return "stylegan2"
class PixelNorm(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)

    if k.ndim == 1:
        k = k[None, :] * k[:, None]

    k /= k.sum()

    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2 + factor - 1
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)

        return out


class Downsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()

        self.factor = factor
        kernel = make_kernel(kernel)
        self.register_buffer('kernel', kernel)

        p = kernel.shape[0] - factor

        pad0 = (p + 1) // 2
        pad1 = p // 2

        self.pad = (pad0, pad1)

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, up=1, down=self.factor, pad=self.pad)

        return out


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()

        kernel = make_kernel(kernel)

        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)

        self.register_buffer('kernel', kernel)

        self.pad = pad

    def forward(self, input):
        out = upfirdn2d(input, self.kernel, pad=self.pad)

        return out


class EqualConv2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super().__init__()

        self.weight = nn.Parameter(
            torch.randn(out_channel, in_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]},'
            f' {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})'
        )


class EqualLinear(nn.Module):
    def __init__(
        self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init))

        else:
            self.bias = None

        self.activation = activation

        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            out = fused_leaky_relu(out, self.bias * self.lr_mul)

        else:
            out = F.linear(
                input, self.weight * self.scale, bias=self.bias * self.lr_mul
            )

        return out

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.weight.shape[1]}, {self.weight.shape[0]})'
        )


class ScaledLeakyReLU(nn.Module):
    def __init__(self, negative_slope=0.2):
        super().__init__()

        self.negative_slope = negative_slope

    def forward(self, input):
        out = F.leaky_relu(input, negative_slope=self.negative_slope)

        return out * math.sqrt(2)


class ModulatedConv2d(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        demodulate=True,
        upsample=False,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
    ):
        super().__init__()

        self.eps = 1e-8
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample
        self.downsample = downsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            self.blur = Blur(blur_kernel, pad=(pad0, pad1))

        fan_in = in_channel * kernel_size ** 2
        self.scale = 1 / math.sqrt(fan_in)
        self.padding = kernel_size // 2

        self.weight = nn.Parameter(
            torch.randn(1, out_channel, in_channel, kernel_size, kernel_size) # [1, 512, 512, 3, 3]
        )

        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)

        self.demodulate = demodulate

    def __repr__(self):
        return (
            f'{self.__class__.__name__}({self.in_channel}, {self.out_channel}, {self.kernel_size}, '
            f'upsample={self.upsample}, downsample={self.downsample})'
        )

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape # [bs, 512, 4, 4]

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1) # [bs, 1, 512, 1, 1]
        # broadcast self.weights and style and element-wise multiply with self.weight 
        weight = self.scale * self.weight * style # Eq.1: [bs, 512, 512, 3, 3]

        if self.demodulate: # Eq.3
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(
            batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size # [bs*512, 512, 3, 3]
        )

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(
                batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size
            )
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size
            )
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)

        elif self.downsample:
            input = self.blur(input)
            _, _, height, width = input.shape
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        else:
            # print(input.device, input.is_contiguous())
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()

        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()

        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()

        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        batch = input.shape[0]
        out = self.input.repeat(batch, 1, 1, 1)

        return out


class StyledConv(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        style_dim,
        upsample=False,
        blur_kernel=[1, 3, 3, 1],
        demodulate=True,
    ):
        super().__init__()

        self.conv = ModulatedConv2d(
            in_channel,
            out_channel,
            kernel_size,
            style_dim,
            upsample=upsample,
            blur_kernel=blur_kernel,
            demodulate=demodulate,
        )

        self.noise = NoiseInjection()
        # self.bias = nn.Parameter(torch.zeros(1, out_channel, 1, 1))
        # self.activate = ScaledLeakyReLU(0.2)
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        # out = out + self.bias
        out = self.activate(out)

        return out


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=[1, 3, 3, 1]):
        super().__init__()

        if upsample:
            self.upsample = Upsample(blur_kernel)

        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style)
        out = out + self.bias

        if skip is not None:
            skip = self.upsample(skip)

            out = out + skip

        return out


class SimpleLabelEncoder(nn.Module):
    def __init__(self, size=256, input_dim=14, embed_dim=256):
        super().__init__()
        self.log_size = int(math.log(size, 2)) # if size=256, log_size=8
        self.fc = nn.Linear(input_dim, embed_dim)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x = [BS, input_dim]
        out = self.relu(self.fc(x)) # [BS, embed_dim]
        return out.unsqueeze(1).repeat(1, self.log_size-1, 1) # [BS, log_size-1, embed_dim]


class LabelEncoder(nn.Module):
    def __init__(self, size=64, input_dim=14, embed_dim=256):
        super().__init__()
        self.log_size = int(math.log(size, 2)) # size=256, log_size=8
        for i in range(2, self.log_size+1):
            self.add_module(f'fc{i}', nn.Linear(input_dim, embed_dim))
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # x = [BS, input_dim]
        outs = []
        for i in range(2, self.log_size+1):
            module = self._modules[f'fc{i}']
            out = self.relu(module(x)) # [BS, embed_dim]
            outs.append(out)
        return torch.stack(outs).transpose(1,0) # [BS, log_size-1, embed_dim]


class Generator(nn.Module):
    def __init__(
        self,
        size=256,
        embed_dim=256,
        z_dim=64,
        n_mlp=8,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
    ):
        super().__init__()

        self.size = size
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.log_size = int(math.log(size, 2)) # if size=256: log_size=8
        self.n_mlp = n_mlp
        latent_dim = embed_dim + z_dim
        # MLPs
        if n_mlp>0:
            layers = [PixelNorm()]
            for i in range(n_mlp):
                layers.append(EqualLinear(z_dim, z_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
            self.mlp = nn.Sequential(*layers)

        # GAN
        self.channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, latent_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], latent_dim, upsample=False)

        self.num_layers = (self.log_size - 2) * 2 + 1 # if size=256: log_size=8, num_layers=13

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers): # 0,...,13
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1): # 3, 4, 5, 6, 7, 8

            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    latent_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, latent_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, latent_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2 # if size=256, log_size=8, n_latent=14

    def make_noise(self):
        device = self.input.input.device
        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]
        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(n_latent, self.z_dim, device=self.input.input.device)
        latent = self.mlp(latent_in).mean(0, keepdim=True)
        return latent

    def forward(
        self,
        label_embedding,
        z,
        return_latents=False,
        truncation=1,
        truncation_latent=None,
        noise=None,
        randomize_noise=True,
    ):
        bs = label_embedding.shape[0]
        device = label_embedding.device
        if z is None:
            z = torch.randn(bs, self.z_dim).to(device)
        if self.n_mlp > 0:
            # MLP
            w = self.mlp(z) # [BS, z_dim]
        else:
            # no MLP
            w = z # [BS, z_dim]

        # truncation
        if truncation < 1: # truncation in W space
            w = truncation_latent + truncation * (w - truncation_latent)

        # repeat w because we need inject twice per resolution
        repeated_w = w.unsqueeze(1).repeat(1, self.log_size-1, 1) # if size=256: repeated_w = [BS, 7, z_dim]
        # if size=256, label_embedding = [BS, 7, embed_dim]
        latent = torch.cat([label_embedding, repeated_w], dim=-1) # if size=256, latent = [BS, 7, embed_dim+z_dim]

        # GAN
        # create noise for the Synthesis Network
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers #  if size=64: [None] * 9
            else:
                noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]

        out = self.input(latent) # [BS, 512, 4, 4]
        out = self.conv1(out, latent[:, 0], noise=noise[0]) # latent[:, 0] => 4x4
        skip = self.to_rgb1(out, latent[:, 0]) # latent[:, 1] => 4x4
        # print(f'skip.shape={skip.shape}')

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            i += 1
            out = conv2(out, latent[:, i], noise=noise2)
            skip = to_rgb(out, latent[:, i], skip)
            # print(f'i={i}, skip.shape={skip.shape}')
            i += 1

        image = skip
        if return_latents:
            return image, latent
        else:
            return image, None
    def get_latent(self,label_embedding,z,truncation=1):
        bs = label_embedding.shape[0]
        device = label_embedding.device
        if z is None:
            z = torch.randn(bs, self.z_dim).to(device)
        if self.n_mlp > 0:
            # MLP
            w = self.mlp(z) # [BS, z_dim]
        else:
            # no MLP
            w = z # [BS, z_dim]

        # truncation
        if truncation < 1: # truncation in W space
            w =  truncation * w

        # repeat w because we need inject twice per resolution
        repeated_w = w.unsqueeze(1).repeat(1, self.log_size-1, 1) # if size=256: repeated_w = [BS, 7, z_dim]
        # if size=256, label_embedding = [BS, 7, embed_dim]
        latent = torch.cat([label_embedding, repeated_w], dim=-1) # if size=256, latent = [BS, 7, embed_dim+z_dim]
        return latent
    def synthesis(
        self,
        latent,
        return_latents=False,
        noise=None,
        randomize_noise=True,
    ):
        bs = latent.shape[0]
        device = latent.device

        # GAN
        # create noise for the Synthesis Network
        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers #  if size=64: [None] * 9
            else:
                noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]

        out = self.input(latent) # [BS, 512, 4, 4]
        out = self.conv1(out, latent[:, 0], noise=noise[0]) # latent[:, 0] => 4x4
        skip = self.to_rgb1(out, latent[:, 0]) # latent[:, 1] => 4x4
        # print(f'skip.shape={skip.shape}')

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            i += 1
            out = conv2(out, latent[:, i], noise=noise2)
            skip = to_rgb(out, latent[:, i], skip)
            # print(f'i={i}, skip.shape={skip.shape}')
            i += 1

        image = skip
        if return_latents:
            return image, latent
        else:
            return image, None

class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        downsample=False,
        blur_kernel=[1, 3, 3, 1],
        bias=True,
        activate=True,
    ):
        layers = []

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2
            self.padding = 0

        else:
            stride = 1
            self.padding = kernel_size // 2

        layers.append(
            EqualConv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=self.padding,
                stride=stride,
                bias=bias and not activate,
            )
        )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super().__init__(*layers)


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.conv1 = ConvLayer(in_channel, in_channel, 3)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)

        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)

        return out


class ConditionalResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, style_dim):
        super().__init__()
        self.conv1 = ModulatedConv2d(in_channel, in_channel, 3, style_dim)
        self.conv2 = ConvLayer(in_channel, out_channel, 3, downsample=True)
        self.skip = ConvLayer(
            in_channel, out_channel, 1, downsample=True, activate=False, bias=False
        )

    def forward(self, input, style):
        out = self.conv1(input, style)
        out = self.conv2(out)
        skip = self.skip(input)
        out = (out + skip) / math.sqrt(2)
        return out


class Discriminator(nn.Module):
    def __init__(
        self, 
        size=256, 
        embed_dim=256,
        z_dim=64,
        channel_multiplier=2):

        super().__init__()
        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        latent_dim = embed_dim
        # for unconditional conv
        convs = [ConvLayer(3, channels[size], 1)] # convs has 1 layer
        log_size = int(math.log(size, 2))
        in_channel = channels[size]
        for i in range(log_size, 2, -1): # if size=256, log_size=8: 8,7,6,5,4,3
            out_channel = channels[2 ** (i - 1)]
            convs.append(ResBlock(in_channel, out_channel))
            in_channel = out_channel
        self.uncond_convs = nn.Sequential(*convs)

        # for conditional conv
        cond_convs = [ModulatedConv2d(3, channels[size], 1, latent_dim)] # convs has 1 layer
        log_size = int(math.log(size, 2))
        in_channel = channels[size]
        for i in range(log_size, 2, -1): # log_size=8: 8,7,6,5,4,3 => convs has 5 layers after loop finishes
            out_channel = channels[2 ** (i - 1)]
            cond_convs.append(ConditionalResBlock(in_channel, out_channel, latent_dim))
            in_channel = out_channel
        self.cond_convs = nn.Sequential(*cond_convs)

        # minibatch stddev
        self.stddev_group = 4
        # self.stddev_group = 2
        self.stddev_feat = 1

        # final layers
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )

    def forward(self, img, label_embedding=None):
        bs = img.shape[0]
        assert bs%self.stddev_group == 0, 'batch_size has to be self.stddev_group multipliers'

        if label_embedding is None:
            out = self.uncond_convs(img) # [2,512,4,4]
        else:
            out = img
            label_embedding = label_embedding.flip(1)
            i = 0
            # conditioned at each scale
            for conv in self.cond_convs:
                style = label_embedding[:,i]
                out = conv(out, style)
                i += 1
        
        batch, channel, height, width = out.shape
        group = min(batch, self.stddev_group)
        stddev = out.view(
            group, -1, self.stddev_feat, channel // self.stddev_feat, height, width
        )
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8) # compute stddev for each group
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2) # average among pixels
        stddev = stddev.repeat(group, 1, height, width) # repeat for every sample in each group 
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


if __name__ == '__main__':
    import common
    import pdb
    size = 256
    device = 'cuda'
    bs = 8
    embed_dim = 256
    z_dim = 256
    n_mlp = 8

    # dummy input
    label = torch.randn(bs, 14).to(device)
    z = torch.randn(bs, z_dim).to(device)
    img = torch.randn(bs, 3, size, size).to(device)

    # Label Encoder
    enc = LabelEncoder(size=size, input_dim=14, embed_dim=embed_dim).to(device)
    print('LabelEncoder', common.count_parameters(enc))
    if device == 'cuda':
        enc = nn.DataParallel(enc)
    label_embedding = enc(label)
    print('label_embedding:', label_embedding.shape)

    enc = SimpleLabelEncoder(size=size, input_dim=14, embed_dim=embed_dim).to(device)
    print('LabelEncoder', common.count_parameters(enc))
    if device == 'cuda':
        enc = nn.DataParallel(enc)
    simple_label_embedding = enc(label)
    print('simple_label_embedding:', simple_label_embedding.shape)

    # test Generator
    g = Generator(size=size, embed_dim=embed_dim, z_dim=z_dim, n_mlp=n_mlp).to(device)
    print('Generator', common.count_parameters(g))
    if device == 'cuda':
        g = nn.DataParallel(g)
    img, _ = g(label_embedding, z, return_latents=True)
    print('G output', img.shape)

    # test Discriminator
    d = Discriminator(size=size, embed_dim=embed_dim, z_dim=z_dim, channel_multiplier=2).to(device)
    print('Discrminator', common.count_parameters(d))
    if device == 'cuda':
        d = nn.DataParallel(d)
    out = d(img, label_embedding)
    print('Conditonal D output', out.shape)

    out = d(img)
    print('Unconditonal D output', out.shape)

    pdb.set_trace()