# Fast Fourier Convolution NeurIPS 2020
# original implementation https://github.com/pkumivision/FFC/blob/main/model_zoo/ffc.py
# paper https://proceedings.neurips.cc/paper/2020/file/2fd5d41ec6cfab47e32164d5624269b1-Paper.pdf

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from guided_diffusion.modules.base import get_activation, BaseDiscriminator
from guided_diffusion.modules.spatial_transform import LearnableSpatialTransformWrapper
from guided_diffusion.modules.squeeze_excitation import SELayer
from guided_diffusion.modules.utils import get_shape

import torch as th
import math
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = th.exp(
        -math.log(max_period) * th.arange(start=0, end=half, dtype=th.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = th.cat([th.cos(args), th.sin(args)], dim=-1)
    if dim % 2:
        embedding = th.cat([embedding, th.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module

class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x


class FFCSE_block(nn.Module):

    def __init__(self, channels, ratio_g):
        super(FFCSE_block, self).__init__()
        in_cg = int(channels * ratio_g)
        in_cl = channels - in_cg
        r = 16

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(channels, channels // r,
                               kernel_size=1, bias=True)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_a2l = None if in_cl == 0 else nn.Conv2d(
            channels // r, in_cl, kernel_size=1, bias=True)
        self.conv_a2g = None if in_cg == 0 else nn.Conv2d(
            channels // r, in_cg, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x if type(x) is tuple else (x, 0)
        id_l, id_g = x

        x = id_l if type(id_g) is int else torch.cat([id_l, id_g], dim=1)
        x = self.avgpool(x)
        x = self.relu1(self.conv1(x))

        x_l = 0 if self.conv_a2l is None else id_l * \
            self.sigmoid(self.conv_a2l(x))
        x_g = 0 if self.conv_a2g is None else id_g * \
            self.sigmoid(self.conv_a2g(x))
        return x_l, x_g


class FourierUnit(nn.Module):

    def __init__(self, in_channels, out_channels, groups=1, spatial_scale_factor=None, spatial_scale_mode='bilinear',
                 spectral_pos_encoding=False, use_se=False, se_kwargs=None, ffc3d=False, fft_norm='ortho'):
        # bn_layer not used
        super(FourierUnit, self).__init__()
        self.groups = groups

        self.conv_layer = torch.nn.Conv2d(in_channels=in_channels * 2 + (2 if spectral_pos_encoding else 0),
                                          out_channels=out_channels * 2,
                                          kernel_size=1, stride=1, padding=0, groups=self.groups, bias=False)
        self.bn = torch.nn.BatchNorm2d(out_channels * 2)
        self.relu = torch.nn.ReLU(inplace=True)

        # squeeze and excitation block
        self.use_se = use_se
        if use_se:
            if se_kwargs is None:
                se_kwargs = {}
            self.se = SELayer(self.conv_layer.in_channels, **se_kwargs)

        self.spatial_scale_factor = spatial_scale_factor
        self.spatial_scale_mode = spatial_scale_mode
        self.spectral_pos_encoding = spectral_pos_encoding
        self.ffc3d = ffc3d
        self.fft_norm = fft_norm

    def forward(self, x):
        batch = x.shape[0]

        if self.spatial_scale_factor is not None:
            orig_size = x.shape[-2:]
            x = F.interpolate(x, scale_factor=self.spatial_scale_factor, mode=self.spatial_scale_mode, align_corners=False)

        r_size = x.size()
        # (batch, c, h, w/2+1, 2)
        fft_dim = (-3, -2, -1) if self.ffc3d else (-2, -1)
        ffted = torch.fft.rfftn(x, dim=fft_dim, norm=self.fft_norm)
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()  # (batch, c, 2, h, w/2+1)
        ffted = ffted.view((batch, -1,) + ffted.size()[3:])

        if self.spectral_pos_encoding:
            height, width = ffted.shape[-2:]
            coords_vert = torch.linspace(0, 1, height)[None, None, :, None].expand(batch, 1, height, width).to(ffted)
            coords_hor = torch.linspace(0, 1, width)[None, None, None, :].expand(batch, 1, height, width).to(ffted)
            ffted = torch.cat((coords_vert, coords_hor, ffted), dim=1)

        if self.use_se:
            ffted = self.se(ffted)

        ffted = self.conv_layer(ffted)  # (batch, c*2, h, w/2+1)
        ffted = self.relu(self.bn(ffted))

        ffted = ffted.view((batch, -1, 2,) + ffted.size()[2:]).permute(
            0, 1, 3, 4, 2).contiguous()  # (batch,c, t, h, w/2+1, 2)
        ffted = torch.complex(ffted[..., 0], ffted[..., 1])

        ifft_shape_slice = x.shape[-3:] if self.ffc3d else x.shape[-2:]
        output = torch.fft.irfftn(ffted, s=ifft_shape_slice, dim=fft_dim, norm=self.fft_norm)

        if self.spatial_scale_factor is not None:
            output = F.interpolate(output, size=orig_size, mode=self.spatial_scale_mode, align_corners=False)

        return output


class SpectralTransform(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, groups=1, enable_lfu=True, **fu_kwargs):
        # bn_layer not used
        super(SpectralTransform, self).__init__()
        self.enable_lfu = enable_lfu
        if stride == 2:
            self.downsample = nn.AvgPool2d(kernel_size=(2, 2), stride=2)
        else:
            self.downsample = nn.Identity()

        self.stride = stride
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels //
                      2, kernel_size=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels // 2),
            nn.ReLU(inplace=True)
        )
        self.fu = FourierUnit(
            out_channels // 2, out_channels // 2, groups, **fu_kwargs)
        if self.enable_lfu:
            self.lfu = FourierUnit(
                out_channels // 2, out_channels // 2, groups)
        self.conv2 = torch.nn.Conv2d(
            out_channels // 2, out_channels, kernel_size=1, groups=groups, bias=False)

    def forward(self, x):

        x = self.downsample(x)
        x = self.conv1(x)
        output = self.fu(x)

        if self.enable_lfu:
            n, c, h, w = x.shape
            split_no = 2
            split_s = h // split_no
            xs = torch.cat(torch.split(
                x[:, :c // 4], split_s, dim=-2), dim=1).contiguous()
            xs = torch.cat(torch.split(xs, split_s, dim=-1),
                           dim=1).contiguous()
            xs = self.lfu(xs)
            xs = xs.repeat(1, 1, split_no, split_no).contiguous()
        else:
            xs = 0

        output = self.conv2(x + output + xs)

        return output


class FFC(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 ratio_gin, ratio_gout, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, enable_lfu=True,
                 padding_type='reflect', gated=False, **spectral_kwargs):
        super(FFC, self).__init__()

        assert stride == 1 or stride == 2, "Stride should be 1 or 2."
        self.stride = stride

        in_cg = int(in_channels * ratio_gin)
        in_cl = in_channels - in_cg
        out_cg = int(out_channels * ratio_gout)
        out_cl = out_channels - out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.ratio_gin = ratio_gin
        self.ratio_gout = ratio_gout
        self.global_in_num = in_cg

        module = nn.Identity if in_cl == 0 or out_cl == 0 else nn.Conv2d
        self.convl2l = module(in_cl, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cl == 0 or out_cg == 0 else nn.Conv2d
        self.convl2g = module(in_cl, out_cg, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cl == 0 else nn.Conv2d
        self.convg2l = module(in_cg, out_cl, kernel_size,
                              stride, padding, dilation, groups, bias, padding_mode=padding_type)
        module = nn.Identity if in_cg == 0 or out_cg == 0 else SpectralTransform
        self.convg2g = module(
            in_cg, out_cg, stride, 1 if groups == 1 else groups // 2, enable_lfu, **spectral_kwargs)

        self.gated = gated
        module = nn.Identity if in_cg == 0 or out_cl == 0 or not self.gated else nn.Conv2d
        self.gate = module(in_channels, 2, 1)

    def forward(self, x):
        x_l, x_g = x if type(x) is tuple else (x, 0)
        out_xl, out_xg = 0, 0

        if self.gated:
            total_input_parts = [x_l]
            if torch.is_tensor(x_g):
                total_input_parts.append(x_g)
            total_input = torch.cat(total_input_parts, dim=1)

            gates = torch.sigmoid(self.gate(total_input))
            g2l_gate, l2g_gate = gates.chunk(2, dim=1)
        else:
            g2l_gate, l2g_gate = 1, 1

        if self.ratio_gout != 1:
            out_xl = self.convl2l(x_l) + self.convg2l(x_g) * g2l_gate
        if self.ratio_gout != 0:
            out_xg = self.convl2g(x_l) * l2g_gate + self.convg2g(x_g)

        return out_xl, out_xg


class FFC_BN_ACT(TimestepBlock):

    def __init__(self, in_channels, out_channels, emb_channels,
                 kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0, dilation=1, groups=1, bias=False,
                 norm_layer=nn.BatchNorm2d, activation_layer=nn.Identity,
                 padding_type='reflect',
                 enable_lfu=True, **kwargs):
        super(FFC_BN_ACT, self).__init__()


        self.out_cg = int(out_channels * ratio_gout)
        self.out_cl = out_channels - self.out_cg
        #groups_g = 1 if groups == 1 else int(groups * ratio_gout)
        #groups_l = 1 if groups == 1 else groups - groups_g

        self.emb_channels = emb_channels
        # 修改 Linear 的输出维度，我们需要 4 组参数：
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_cl + 2 * self.out_cg,  # 总共 2 * out_channels
            ),
        )
        self.ffc = FFC(in_channels, out_channels, kernel_size,
                       ratio_gin, ratio_gout, stride, padding, dilation,
                       groups, bias, enable_lfu, padding_type=padding_type, **kwargs)
        lnorm = nn.Identity if ratio_gout == 1 else norm_layer
        gnorm = nn.Identity if ratio_gout == 0 else norm_layer
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = lnorm(out_channels - global_channels)
        self.bn_g = gnorm(global_channels)

        lact = nn.Identity if ratio_gout == 1 else activation_layer
        gact = nn.Identity if ratio_gout == 0 else activation_layer
        self.act_l = lact(inplace=True)
        self.act_g = gact(inplace=True)
        nn.init.zeros_(self.emb_layers[1].weight)
        nn.init.zeros_(self.emb_layers[1].bias)

    def forward(self, x, emb):
        x_l, x_g = self.ffc(x)

        emb_out = self.emb_layers(emb)
        emb_l, emb_g = torch.split(emb_out, [2 * self.out_cl, 2 * self.out_cg], dim=1)

        scale_l, shift_l = torch.chunk(emb_l, 2, dim=1)
        scale_g, shift_g = torch.chunk(emb_g, 2, dim=1)

        x_l = self.act_l(self.bn_l(x_l)* (1 + scale_l) + shift_l)
        x_g = self.act_g(self.bn_g(x_g)* (1 + scale_g) + shift_g)
        return x_l, x_g

class FFCResnetBlock(TimestepBlock):
    def __init__(self, dim, emb_channels,padding_type, norm_layer, activation_layer=nn.ReLU, dilation=1,
                 spatial_transform_kwargs=None, inline=False):
        super().__init__()
        self.conv1 = FFC_BN_ACT(dim, dim, emb_channels,kernel_size=3, ratio_gin=0.75, ratio_gout=0.75,stride=1,padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                enable_lfu=False)
        self.conv2 = FFC_BN_ACT(dim, dim, emb_channels,kernel_size=3,ratio_gin=0.75, ratio_gout=0.75,stride=1, padding=dilation, dilation=dilation,
                                norm_layer=norm_layer,
                                activation_layer=activation_layer,
                                padding_type=padding_type,
                                enable_lfu=False)
        if spatial_transform_kwargs is not None:
            self.conv1 = LearnableSpatialTransformWrapper(self.conv1, **spatial_transform_kwargs)
            self.conv2 = LearnableSpatialTransformWrapper(self.conv2, **spatial_transform_kwargs)
        self.inline = inline

    def forward(self, x, emb):
        if self.inline:
            x_l, x_g = x[:, :-self.conv1.ffc.global_in_num], x[:, -self.conv1.ffc.global_in_num:]
        else:
            x_l, x_g = x if type(x) is tuple else (x, 0)

        id_l, id_g = x_l, x_g

        x_l, x_g = self.conv1((x_l, x_g),emb)
        x_l, x_g = self.conv2((x_l, x_g),emb)

        x_l, x_g = id_l + x_l, id_g + x_g
        out = x_l, x_g
        if self.inline:
            out = torch.cat(out, dim=1)
        return out


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        assert isinstance(x, tuple)
        x_l, x_g = x
        assert torch.is_tensor(x_l) or torch.is_tensor(x_g)
        if not torch.is_tensor(x_g):
            return x_l
        return torch.cat(x, dim=1)


class FFCResNetGenerator(nn.Module):
    def __init__(self, input_nc=3, ngf=64,n_downsampling=3, n_blocks=18, norm_layer=nn.BatchNorm2d,
                 padding_type='reflect', activation_layer=nn.ReLU,
                 spatial_transform_layers=None, spatial_transform_kwargs={},
                 max_features=1024):
        assert (n_blocks >= 0)
        super().__init__()

        self.input_hint_block = nn.Sequential(
            nn.Conv2d(input_nc, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d( 16, 16, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d( 16, 32, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d( 32, 32, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d( 32, 96, 3, padding=1, stride=2),
            nn.SiLU(),
            nn.Conv2d( 96, 96, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d( 96, 256, 3, padding=1, stride=2),
            nn.SiLU(),
            zero_module(nn.Conv2d( 256, ngf, 3, padding=1))
        )

        self.model_channels=ngf
        time_embed_dim = ngf * 4
        self.time_embed = nn.Sequential(
            nn.Linear(ngf, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        layers=[nn.ReflectionPad2d(3),
                 FFC_BN_ACT(input_nc, ngf, time_embed_dim,kernel_size=7, ratio_gin=0, ratio_gout=0,stride=1,padding=0, norm_layer=norm_layer,
                            activation_layer=activation_layer, padding_type='reflect', enable_lfu=False)]
        self.model = nn.ModuleList(TimestepEmbedSequential(layers))
        self.zero_convs = nn.ModuleList([self.make_zero_conv(ngf)])

        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            if i == n_downsampling - 1:
                ratio_gin, ratio_gout = 0.75,0.75
            else:
                ratio_gin,ratio_gout=0,0
            self.model += [FFC_BN_ACT(min(max_features, ngf * mult),
                                 min(max_features, ngf * mult * 2),time_embed_dim,
                                 kernel_size=3, ratio_gin=ratio_gin, ratio_gout=ratio_gout,stride=2, padding=1,
                                 norm_layer=norm_layer,
                                 activation_layer=activation_layer,
                                 padding_type='reflect', enable_lfu=False)]
            self.zero_convs.append(self.make_zero_conv(min(max_features, ngf * mult * 2)))

        mult = 2 ** n_downsampling
        feats_num_bottleneck = min(max_features, ngf * mult)

        ### resnet blocks
        self.model += FFCResnetBlock(feats_num_bottleneck, time_embed_dim, padding_type=padding_type,
                                 activation_layer=activation_layer,
                                 norm_layer=norm_layer)
        self.zero_convs.append(self.make_zero_conv(feats_num_bottleneck))
        layers=[]
        for i in range(n_blocks-1):
            layers += FFCResnetBlock(feats_num_bottleneck, time_embed_dim,padding_type=padding_type, activation_layer=activation_layer,
                                          norm_layer=norm_layer)
            if spatial_transform_layers is not None and i in spatial_transform_layers:
                cur_resblock = LearnableSpatialTransformWrapper(cur_resblock, **spatial_transform_kwargs)
        self.model += [TimestepEmbedSequential(layers)]
        self.zero_convs.append(self.make_zero_conv(feats_num_bottleneck))


    def make_zero_conv(self, channels):
        return nn.Sequential(zero_module(nn.Conv2d(channels, channels, 1, padding=0)))

    def forward(self, z, timesteps):
        emb = self.time_embed(timestep_embedding(
            timesteps, self.model_channels))
        guided_hint = self.input_hint_block(z)

        outs = []

        for module, zero_conv in zip(self.model, self.zero_convs):
            if guided_hint is not None:
                z = module(z, emb)
                z_l, z_g = z  # 第一层返回的一定是 Tuple
                z = (z_l + guided_hint, z_g)
                guided_hint = None
            else:
                z = module(z, emb)
            outs.append(zero_conv(ConcatTupleLayer(z)))
        return outs
