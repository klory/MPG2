import math
import random

import torch
from torch import nn
from torch.nn import functional as F

import sys
sys.path.append('../')
from mpg.op import FusedLeakyReLU, fused_leaky_relu, upfirdn2d

class OneLabelEncoder(nn.Module):
    def __init__(self, n_layers=1, input_dim=13, embed_dim=256):
        super().__init__()
        main = []
        for i in range(n_layers):
            in_dim = input_dim if i==0 else embed_dim
            main.append(nn.Linear(in_dim, embed_dim))
            main.append(nn.ReLU())
        self.main = nn.Sequential(*main)
        # print(self.main)

    def forward(self, x):
        # x = [BS, input_dim]
        out = self.main(x) # [BS, embed_dim]
        return out


class LabelEncoder(nn.Module):
    def __init__(self, size=64, input_dim=13, embed_dim=256, n_layers=1, type_='many'):
        super().__init__()
        self.log_size = int(math.log(size, 2)) # size=64, log_size=6
        assert type_ in ['one', 'many']
        if type_ == 'one':
            self.add_module(f'enc', OneLabelEncoder(n_layers, input_dim, embed_dim))
        elif type_ == 'many':
            for i in range(2, self.log_size+1):
                self.add_module(f'enc{i}', OneLabelEncoder(n_layers, input_dim, embed_dim))
        self.type = type_

    def forward(self, x):
        # x = [BS, input_dim]
        if self.type == 'one':
            module = self._modules[f'enc']
            out = module(x) # [BS, embed_dim]
            return out.unsqueeze(1).repeat(1, self.log_size-1, 1) # [BS, log_size-1, embed_dim]
        elif self.type == 'many':
            outs = []
            for i in range(2, self.log_size+1):
                module = self._modules[f'enc{i}']
                out = module(x) # [BS, embed_dim]
                outs.append(out)
            return torch.stack(outs).transpose(1,0) # [BS, log_size-1, embed_dim]


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


class Generator(nn.Module):
    def __init__(
        self,
        size=64,
        embed_dim=256,
        style_dim=256,
        n_mlp=8,
        channel_multiplier=2,
        blur_kernel=[1, 3, 3, 1],
        lr_mlp=0.01,
        cat_z=False,
    ):
        super().__init__()

        self.log_size = int(math.log(size, 2)) # size=64, log_size=6

        # MLP
        self.size = size

        if cat_z:
            style_dim = style_dim + embed_dim
            embed_dim = 0

        self.style_dim = style_dim

        layers = [PixelNorm()]

        for i in range(n_mlp):
            layers.append(
                EqualLinear(
                    style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'
                )
            )

        self.style = nn.Sequential(*layers)

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
        self.conv1 = StyledConv(
            self.channels[4], self.channels[4], 3, embed_dim+style_dim, blur_kernel=blur_kernel
        )
        self.to_rgb1 = ToRGB(self.channels[4], embed_dim+style_dim, upsample=False)

        self.num_layers = (self.log_size - 2) * 2 + 1 # size=64, log_size=6, num_layers=9

        self.convs = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        in_channel = self.channels[4]

        for layer_idx in range(self.num_layers): # 0,...,8
            res = (layer_idx + 5) // 2
            shape = [1, 1, 2 ** res, 2 ** res]
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(*shape))

        for i in range(3, self.log_size + 1): # 3, 4, 5, 6

            out_channel = self.channels[2 ** i]

            self.convs.append(
                StyledConv(
                    in_channel,
                    out_channel,
                    3,
                    embed_dim+style_dim,
                    upsample=True,
                    blur_kernel=blur_kernel,
                )
            )

            self.convs.append(
                StyledConv(
                    out_channel, out_channel, 3, embed_dim+style_dim, blur_kernel=blur_kernel
                )
            )

            self.to_rgbs.append(ToRGB(out_channel, embed_dim+style_dim))

            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2 # log_size=6, n_latent=10

    def make_noise(self):
        device = self.input.input.device

        noises = [torch.randn(1, 1, 2 ** 2, 2 ** 2, device=device)]

        for i in range(3, self.log_size + 1):
            for _ in range(2):
                noises.append(torch.randn(1, 1, 2 ** i, 2 ** i, device=device))

        return noises

    def mean_latent(self, n_latent):
        latent_in = torch.randn(
            n_latent, self.style_dim, device=self.input.input.device
        )
        latent = self.style(latent_in).mean(0, keepdim=True)

        return latent

    def get_latent(self, input):
        return self.style(input)

    def forward(
        self,
        styles,
        label_embedding=None,
        return_latents=False,
        inject_index=None,
        truncation=1,
        truncation_latent=None,
        input_is_latent=False,
        noise=None,
        randomize_noise=True,
    ):
        # MLP
        if not input_is_latent:
            if label_embedding is not None and label_embedding.ndim == 2:
                styles = [torch.cat([s, label_embedding], dim=1) for s in styles]
            styles = [self.style(s) for s in styles]
        # now styles=[N, 512],...

        if noise is None:
            if randomize_noise:
                noise = [None] * self.num_layers #  [None] * 9
            else:
                noise = [
                    getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
                ]

        if truncation < 1: # truncation in W space
            style_t = []

            for style in styles:
                style_t.append(
                    truncation_latent + truncation * (style - truncation_latent)
                )

            styles = style_t

        if len(styles) < 2: # no style mixing
            inject_index = self.n_latent # size=64, n_latent=10

            if styles[0].ndim < 3:
                latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1) # [N, 10, 512]

            else:
                latent = styles[0]

        else:  # style mixing
            if inject_index is None:
                inject_index = random.randint(1, self.n_latent - 1) # [1,8]

            latent = styles[0].unsqueeze(1).repeat(1, inject_index, 1)
            latent2 = styles[1].unsqueeze(1).repeat(1, self.n_latent - inject_index, 1)

            latent = torch.cat([latent, latent2], 1)

        if label_embedding is not None and label_embedding.ndim == 3:
            # combine label_embedding [bs, n_latent/2, embed_dim] and latent [bs, n_latent, style_dim]
            bs, half_n_latent, embed_dim = label_embedding.shape
            repeated_label_embedding = label_embedding.repeat(1,1,2).view(bs, self.n_latent, self.style_dim)
            if repeated_label_embedding.shape != latent.shape:
                print(bs, half_n_latent, embed_dim, repeated_label_embedding.shape, latent.shape)
            assert repeated_label_embedding.shape == latent.shape, 'repeated_label_embedding and latent have to be the same shape'
            latent = torch.cat([repeated_label_embedding, latent], dim=-1) # [bs, n_latent, embed_dim+style_dim]
        
        # GAN
        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0]) # latent[:, 0] => 4x4

        skip = self.to_rgb1(out, latent[:, 1]) # latent[:, 1] => 4x4

        # i = 1
        i = 2
        for conv1, conv2, noise1, noise2, to_rgb in zip(
            self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs
        ):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            # skip = to_rgb(out, latent[:, i + 2], skip)
            skip = to_rgb(out, latent[:, i + 1], skip)

            i += 2

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
    def __init__(self, size, style_dim, channel_multiplier=2):
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

        # for unconditional conv
        convs = [ConvLayer(3, channels[size], 1)] # convs has 1 layer

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1): # log_size=6: 6,5,4,3 => convs has 5 layers after loop finishes
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        # for conditional conv
        cond_convs = [ModulatedConv2d(3, channels[size], 1, style_dim)] # convs has 1 layer

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1): # log_size=6: 6,5,4,3 => convs has 5 layers after loop finishes
            out_channel = channels[2 ** (i - 1)]

            cond_convs.append(ConditionalResBlock(in_channel, out_channel, style_dim))

            in_channel = out_channel

        self.cond_convs = nn.Sequential(*cond_convs)

        # minibatch stddev
        self.stddev_group = 4
        self.stddev_feat = 1

        # final layers
        self.final_conv = ConvLayer(in_channel + 1, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation='fused_lrelu'),
            EqualLinear(channels[4], 1),
        )
        self.log_size = log_size

    def forward(self, img, label_embedding=None):
        if label_embedding is None:
            out = self.convs(img) # [2,512,4,4]
        else:
            out = img
            if label_embedding.ndim == 3:
                label_embedding = label_embedding.flip(1) # [BS, log_size-1, 256]
            else:
                label_embedding = label_embedding.unsqueeze(1).repeat(1, self.log_size-1, 1) # [BS, log_size-1, embed_dim]
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
        stddev = torch.sqrt(stddev.var(0, unbiased=False) + 1e-8)
        stddev = stddev.mean([2, 3, 4], keepdims=True).squeeze(2)
        stddev = stddev.repeat(group, 1, height, width)
        out = torch.cat([out, stddev], 1)

        out = self.final_conv(out)

        out = out.view(batch, -1)
        out = self.final_linear(out)

        return out


if __name__ == '__main__':
    import common
    import pdb
    res = 256
    style_dim = 256
    embed_dim = 256
    num_categories = 8
    device = 'cuda'

    # dummy input
    txt = [
        'Fresh basil, Pepperoni', 
        'Tomatos',
        '',
        'Arugula, Mushrooms, Black olives'
    ]
    bs = len(txt)
    img = torch.randn(bs, 3, res, res).to(device)
    wrong_img = torch.randn(bs, 3, res, res).to(device)
    binary_label = torch.zeros(bs, num_categories).to(device)
    binary_label[0,0] = 1
    binary_label[0,4] = 1
    binary_label[1,3] = 1
    binary_label[3,1] = 1
    binary_label[3,5] = 1
    binary_label[3,7] = 1
    styles = [torch.randn(bs, style_dim).to(device)]

    # test uncond Generator
    g = Generator(size=res, embed_dim=0, style_dim=style_dim, n_mlp=8).to(device)
    if device == 'cuda':
        g = nn.DataParallel(g)
    img, _ = g(styles, return_latents=True)
    print('Unconditional:', img.shape)

    # test uncond Discriminator
    d = Discriminator(size=res, style_dim=style_dim).to(device)
    if device == 'cuda':
        d = nn.DataParallel(d)
    out = d(img)
    print('Unconditional:', out.shape)
    
    # test cond Generator
    enc = LabelEncoder(
        size=res, input_dim=num_categories, 
        embed_dim=embed_dim, n_layers=2, type_='many').to(device)
    common.count_parameters(enc)
    if device == 'cuda':
        enc = nn.DataParallel(enc)
    txt_embedding = enc(binary_label)
    print('txt_embedding:', txt_embedding.shape)

    g = Generator(size=res, embed_dim=embed_dim, style_dim=style_dim, n_mlp=8, cat_z=False).to(device)
    if device == 'cuda':
        g = nn.DataParallel(g)    
    img, _ = g(styles, txt_embedding, return_latents=True)
    print('Conditional:', img.shape)

    # test cond Discriminator
    d = Discriminator(size=res, style_dim=style_dim).to(device)
    if device == 'cuda':
        d = nn.DataParallel(d)
    out = d(img, txt_embedding)
    print('Conditional:', out.shape)


    # test cond Generator (cat_z)
    enc = OneLabelEncoder(
        input_dim=num_categories, embed_dim=embed_dim, n_layers=2).to(device)
    common.count_parameters(enc)
    if device == 'cuda':
        enc = nn.DataParallel(enc)
    txt_embedding = enc(binary_label)
    print('txt_embedding (cat_z):', txt_embedding.shape)

    g = Generator(size=res, embed_dim=embed_dim, style_dim=style_dim, n_mlp=8, cat_z=True).to(device)
    if device == 'cuda':
        g = nn.DataParallel(g)   
    styles = [torch.randn(bs, style_dim).to(device)] 
    img, _ = g(styles, txt_embedding, return_latents=True)
    print('Conditional (no Mix, cat_z):', img.shape)

    styles = [torch.randn(bs, style_dim).to(device) for _ in range(2)] 
    img, _ = g(styles, txt_embedding, return_latents=True)
    print('Conditional (Mix, cat_z):', img.shape)

    # pdb.set_trace()