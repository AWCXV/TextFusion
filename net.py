import numpy as np
import os
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from PIL import Image
import torch
import math
import torch.nn as nn
from torch.nn.init import _calculate_fan_in_and_fan_out
from timm.models.layers import to_2tuple, trunc_normal_
import torch.nn.functional as F

#Swintransformer --------------- Begin

class RLN(nn.Module):
    r"""Revised LayerNorm"""

    def __init__(self, dim, eps=1e-5, detach_grad=False):
        super(RLN, self).__init__()
        self.eps = eps
        self.detach_grad = detach_grad

        self.weight = nn.Parameter(torch.ones((1, dim, 1, 1)))
        self.bias = nn.Parameter(torch.zeros((1, dim, 1, 1)))

        self.meta1 = nn.Conv2d(1, dim, 1)
        self.meta2 = nn.Conv2d(1, dim, 1)

        trunc_normal_(self.meta1.weight, std=.02)
        nn.init.constant_(self.meta1.bias, 1)

        trunc_normal_(self.meta2.weight, std=.02)
        nn.init.constant_(self.meta2.bias, 0)

    def forward(self, input):
        mean = torch.mean(input, dim=(1, 2, 3), keepdim=True)
        std = torch.sqrt((input - mean).pow(2).mean(dim=(1, 2, 3), keepdim=True) + self.eps)

        normalized_input = (input - mean) / std

        if self.detach_grad:
            rescale, rebias = self.meta1(std.detach()), self.meta2(mean.detach())
        else:
            rescale, rebias = self.meta1(std), self.meta2(mean)

        out = normalized_input * self.weight + self.bias
        return out, rescale, rebias

class Mlp(nn.Module):
    def __init__(self, network_depth, in_features, hidden_features=None, out_features=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.network_depth = network_depth

        self.mlp = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1),
            nn.ReLU(True),
            nn.Conv2d(hidden_features, out_features, 1)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            gain = (8 * self.network_depth) ** (-1 / 4)
            fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
            std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
            trunc_normal_(m.weight, std=std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size ** 2, C)
    return windows

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def get_relative_positions(window_size):
    coords_h = torch.arange(window_size)
    coords_w = torch.arange(window_size)

    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_positions = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww

    relative_positions = relative_positions.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    relative_positions_log = torch.sign(relative_positions) * torch.log(1. + relative_positions.abs())

    return relative_positions_log

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        relative_positions = get_relative_positions(self.window_size)
        self.register_buffer("relative_positions", relative_positions)
        self.meta = nn.Sequential(
            nn.Linear(2, 256, bias=True),
            nn.ReLU(True),
            nn.Linear(256, num_heads, bias=True)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, ass_qkv):
        B_, N, _ = qkv.shape

        qkv = qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)
        ass_qkv = ass_qkv.reshape(B_, N, 3, self.num_heads, self.dim // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        ass_q, ass_k, ass_v = ass_qkv[0], ass_qkv[1], ass_qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        #text modality -> vision
        ass_q = ass_q * self.scale
        q = q * self.scale
        
        attn = (q @ k.transpose(-2, -1))

        #vision -> text modality
        
        ass_attn = (ass_q @ ass_k.transpose(-2, -1))

        relative_position_bias = self.meta(self.relative_positions)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)
        ass_attn = ass_attn + relative_position_bias.unsqueeze(0)

        attn = self.softmax(attn)
        ass_attn = self.softmax(ass_attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, self.dim)
        ass_x = (ass_attn @ ass_v).transpose(1, 2).reshape(B_, N, self.dim)

        return x, ass_x

class Attention(nn.Module):
    def __init__(self, network_depth, dim, num_heads, window_size, shift_size, use_attn=False, conv_type=None):
        super().__init__()
        self.dim = dim
        self.head_dim = int(dim // num_heads)
        self.num_heads = num_heads

        self.window_size = window_size
        self.shift_size = shift_size

        self.network_depth = network_depth
        self.use_attn = use_attn
        self.conv_type = conv_type

        if self.conv_type == 'Conv':
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect'),
                nn.ReLU(True),
                nn.Conv2d(dim, dim, kernel_size=3, padding=1, padding_mode='reflect')
            )

        if self.conv_type == 'DWConv':
            self.conv = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')
            self.conv_ass = nn.Conv2d(dim, dim, kernel_size=5, padding=2, groups=dim, padding_mode='reflect')

        if self.conv_type == 'DWConv' or self.use_attn:
            self.V = nn.Conv2d(dim, dim, 1)
            self.ass_V = nn.Conv2d(dim, dim, 1)
            self.proj = nn.Conv2d(dim, dim, 1)
            self.proj_ass = nn.Conv2d(dim, dim, 1)

        if self.use_attn:
            self.QK = nn.Conv2d(dim, 2*dim, 1)
            self.ass_QK = nn.Conv2d(dim, 2*dim, 1)
            self.attn = WindowAttention(dim, window_size, num_heads)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            w_shape = m.weight.shape

            if w_shape[0] == self.dim * 2:    # QK
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)
            else:
                gain = (8 * self.network_depth) ** (-1/4)
                fan_in, fan_out = _calculate_fan_in_and_fan_out(m.weight)
                std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
                trunc_normal_(m.weight, std=std)

            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def check_size(self, x, shift=False):
        _, _, h, w = x.size()
        mod_pad_h = (self.window_size - h % self.window_size) % self.window_size
        mod_pad_w = (self.window_size - w % self.window_size) % self.window_size

        if shift:
            x = F.pad(x, (self.shift_size, (self.window_size-self.shift_size+mod_pad_w) % self.window_size,
                          self.shift_size, (self.window_size-self.shift_size+mod_pad_h) % self.window_size), mode='reflect')
        else:
            x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, vision, ass_vision):
        B, C, H, W = vision.shape
        if self.conv_type == 'DWConv' or self.use_attn:
            V = self.V(vision)
            ass_V = self.ass_V(ass_vision);

        if self.use_attn:
            QK = self.QK(vision)
            ass_QK = self.ass_QK(ass_vision)

            QKV = torch.cat([QK, V], dim=1)
            ass_QKV = torch.cat([ass_QK, ass_V], dim=1)

            # shift
            shifted_QKV = self.check_size(QKV, self.shift_size > 0)
            shifted_ass_QKV = self.check_size(ass_QKV, self.shift_size > 0)
            Ht, Wt = shifted_QKV.shape[2:]

            # partition windows
            shifted_QKV = shifted_QKV.permute(0, 2, 3, 1)
            shifted_ass_QKV = shifted_ass_QKV.permute(0, 2, 3, 1)

            qkv = window_partition(shifted_QKV, self.window_size)  # nW*B, window_size**2, C
            ass_qkv = window_partition(shifted_ass_QKV, self.window_size)  # nW*B, window_size**2, C

            attn_windows, ass_attn_windows = self.attn(qkv, ass_qkv)

            # merge windows
            shifted_out = window_reverse(attn_windows, self.window_size, Ht, Wt)  # B H' W' C
            ass_shifted_out = window_reverse(ass_attn_windows, self.window_size, Ht, Wt)  # B H' W' C

            # reverse cyclic shift
            out = shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]
            ass_out = ass_shifted_out[:, self.shift_size:(self.shift_size+H), self.shift_size:(self.shift_size+W), :]

            attn_out = out.permute(0, 3, 1, 2)
            ass_attn_out = out.permute(0, 3, 1, 2)

            if self.conv_type in ['Conv', 'DWConv']:
                conv_out = self.conv(V)
                conv_out_ass = self.conv_ass(ass_V)
                out = self.proj(conv_out + attn_out)
                out_ass = self.proj_ass(conv_out_ass + ass_attn_out)
            else:
                out = self.proj(attn_out)

        else:
            if self.conv_type == 'Conv':
                out = self.conv(vision)                # no attention and use conv, no projection
            elif self.conv_type == 'DWConv':
                out = self.proj(self.conv(V))
        return out, out_ass

class TransformerBlock(nn.Module):
    def __init__(self, network_depth, dim, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, mlp_norm=False,
                 window_size=8, shift_size=0, use_attn=True, conv_type=None):
        super().__init__()
        self.use_attn = use_attn
        self.mlp_norm = mlp_norm

        self.norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.ass_norm1 = norm_layer(dim) if use_attn else nn.Identity()
        self.attn = Attention(network_depth, dim, num_heads=num_heads, window_size=window_size,
                              shift_size=shift_size, use_attn=use_attn, conv_type=conv_type)

        self.norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()
        self.ass_norm2 = norm_layer(dim) if use_attn and mlp_norm else nn.Identity()

        self.mlp = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio))
        self.mlp_ass = Mlp(network_depth, dim, hidden_features=int(dim * mlp_ratio));

    def forward(self, vision, ass_vision):
        identity = vision
        ass_identity = ass_vision
        if self.use_attn: vision, rescale, rebias = self.norm1(vision)
        if self.use_attn: ass_vision, ass_rescale, ass_rebias = self.ass_norm1(ass_vision)

        vision, ass_vision = self.attn(vision, ass_vision)

        if self.use_attn: vision = vision * rescale + rebias
        if self.use_attn: ass_vision = ass_vision * ass_rescale + ass_rebias

        vision = identity + vision
        ass_vision = ass_identity + ass_vision

        identity = vision
        ass_identity = ass_vision

        if self.use_attn and self.mlp_norm: vision, rescale, rebias = self.norm2(vision)
        if self.use_attn and self.mlp_norm: ass_vision, ass_rescale, ass_rebias = self.ass_norm2(ass_vision)

        vision = self.mlp(vision)
        ass_vision = self.mlp_ass(ass_vision);

        if self.use_attn and self.mlp_norm: vision = vision * rescale + rebias
        if self.use_attn and self.mlp_norm: ass_vision = ass_vision * ass_rescale + ass_rebias


        vision = identity + vision
        ass_vision = ass_identity + ass_vision


        return vision,ass_vision

class BasicLayer(nn.Module):
    def __init__(self, network_depth, dim, depth, num_heads, mlp_ratio=4.,
                 norm_layer=nn.LayerNorm, window_size=8,
                 attn_ratio=0., attn_loc='last', conv_type=None):

        super().__init__()
        self.dim = dim
        self.depth = depth

        attn_depth = attn_ratio * depth

        if attn_loc == 'last':
            use_attns = [i >= depth-attn_depth for i in range(depth)]
        elif attn_loc == 'first':
            use_attns = [i < attn_depth for i in range(depth)]
        elif attn_loc == 'middle':
            use_attns = [i >= (depth-attn_depth)//2 and i < (depth+attn_depth)//2 for i in range(depth)]

        self.blocks = nn.ModuleList([
            TransformerBlock(network_depth=network_depth,
                             dim=dim,
                             num_heads=num_heads,
                             mlp_ratio=mlp_ratio,
                             norm_layer=norm_layer,
                             window_size=window_size,
                             shift_size=0 if (i % 2 == 0) else window_size // 2,
                             use_attn=use_attns[i], conv_type=conv_type)
            for i in range(depth)])

    def forward(self, vision, ass_vision):
        for blk in self.blocks:
            vision, ass_vision = blk(vision, ass_vision)
        return vision,ass_vision;

class PatchEmbed(nn.Module):
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = patch_size

        #所谓的embed, 好像就是一个把
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size, stride=patch_size,
                              padding=(kernel_size-patch_size+1)//2, padding_mode='reflect')

    def forward(self, x):
        x = self.proj(x)
        return x

class PatchUnEmbed(nn.Module):
    def __init__(self, patch_size=4, out_chans=3, embed_dim=96, kernel_size=None):
        super().__init__()
        self.out_chans = out_chans
        self.embed_dim = embed_dim

        if kernel_size is None:
            kernel_size = 1

        self.proj = nn.Sequential(
            nn.Conv2d(embed_dim, out_chans*patch_size**2, kernel_size=kernel_size,
                      padding=kernel_size//2, padding_mode='reflect'),
            nn.PixelShuffle(patch_size)
        )

    def forward(self, x):
        x = self.proj(x)
        return x

#Swintransformer --------------- End


class TextCorrespond(nn.Module):
    def __init__(self, dim, text_channel, amplify=8):
        super(TextCorrespond, self).__init__()

        #d = max(int(dim/reduction), 4)
        d = int(dim*amplify);

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp_vis = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, text_channel, 1, bias=False)
        )        
        self.mlp_ir = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, text_channel, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_vis, in_ir, text_features):
        # in_feats: b*c*h*w, text_featurees: 1*512
        x_vis = self.mlp_vis(in_vis);                
        x_ir = self.mlp_ir(in_ir);

        text_features = text_features.view(1,text_features.shape[1],1,1).expand_as(x_ir);
        
        x = x_vis + text_features * x_ir;
        return x;

class VTFusion(nn.Module):
    def __init__(self, dim, height=2, reduction=8):
        super(VTFusion, self).__init__()

        #输入的代融合component个数
        self.height = height
        d = max(int(dim/reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, d, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(d, dim*height, 1, bias=False)
        )

        self.softmax = nn.Softmax(dim=1)

    def forward(self, in_feats):
        B, C, H, W = in_feats[0].shape

        in_feats = torch.cat(in_feats, dim=1)
        in_feats = in_feats.view(B, self.height, C, H, W)

        feats_sum = torch.sum(in_feats, dim=1) # B*C*H*W
        
        attn = self.mlp(self.avg_pool(feats_sum)) # mlp(B*C*1*1)->B*(C*2)*1*1
        attn = self.softmax(attn.view(B, self.height, C, 1, 1))

        out = torch.sum(in_feats*attn, dim=1)
        return out

class TextFusionNet(nn.Module):
    def __init__(self, in_chans=1, out_chans=1, window_size=8,
                 embed_dims=[24, 48, 96, 48, 24],
                 mlp_ratios=[2., 4., 4., 2., 2.],
                 depths=[16, 16, 16, 8, 8],
                 num_heads=[2, 4, 6, 1, 1],
                 attn_ratio=[1/4, 1/2, 3/4, 0, 0],
                 conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'],
                 norm_layer=[RLN, RLN, RLN, RLN, RLN]):
        super(TextFusionNet, self).__init__()

        # setting
        self.patch_size = 4
        self.window_size = window_size
        self.mlp_ratios = mlp_ratios
        text_channels = 512;

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        self.patch_embed2 = PatchEmbed(
            patch_size=1, in_chans=in_chans, embed_dim=embed_dims[0], kernel_size=3)

        # backbone
        self.layer1 = BasicLayer(network_depth=sum(depths), dim=embed_dims[0], depth=depths[0],
                                 num_heads=num_heads[0], mlp_ratio=mlp_ratios[0],
                                 norm_layer=norm_layer[0], window_size=window_size,
                                 attn_ratio=attn_ratio[0], attn_loc='last', conv_type=conv_type[0])

        self.vt_features_fusion = VTFusion(embed_dims[0])

        self.patch_merge1 = PatchEmbed(
            patch_size=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])

        self.fuse_text_image = TextCorrespond(embed_dims[0],text_channels,2);

        # merge non-overlapping patches into image
        self.patch_unembed1 = PatchUnEmbed(
            patch_size=1, out_chans=out_chans, embed_dim=text_channels, kernel_size=3)

        self.ac = nn.Tanh();

        #self.p


    def check_image_size(self, x):
        # NOTE: for I2I test
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size - h % self.patch_size) % self.patch_size
        mod_pad_w = (self.patch_size - w % self.patch_size) % self.patch_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward(self, vis, ir, text_features):
        H, W = vis.shape[2:]
        vision = torch.cat([vis],1);
        vision = self.check_image_size(vision);

        ass_vision = torch.cat([ir],1);
        ass_vision = self.check_image_size(ass_vision);

        vision = self.patch_embed(vision);
        ass_vision = self.patch_embed2(ass_vision);       
        
        x,ass_vision = self.layer1(vision,ass_vision);

        text_fused_features = self.fuse_text_image(x,ass_vision,text_features);

        x = self.patch_unembed1(text_fused_features);
        x = self.ac(x);
        x = x/2+0.5;

        x = x[:, :, :H, :W]
        return x

def TextFusionNet_t():
    return TextFusionNet(
        #embed_dims=[24, 48, 96, 48, 24],
        embed_dims=[24,48],
        #mlp_ratios=[2., 4., 4., 2., 2.],
        mlp_ratios=[2.],
        #depths=[4, 4, 4, 2, 2],
        depths=[1],
        #num_heads=[2, 4, 6, 1, 1],
        num_heads=[2],
        #attn_ratio=[0, 1/2, 1, 0, 0],
        attn_ratio=[1],
        #conv_type=['DWConv', 'DWConv', 'DWConv', 'DWConv', 'DWConv'])
        conv_type=['DWConv'])


