from collections import OrderedDict
import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_, DropPath, to_2tuple

from torch.nn import MultiheadAttention
def conv_3xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (2, stride, stride), (1, 0, 0), groups=groups)


def conv_1xnxn(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (1, kernel_size, kernel_size), (1, stride, stride), (0, 0, 0), groups=groups)


def conv_3xnxn_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv3d(inp, oup, (3, kernel_size, kernel_size), (1, stride, stride), (1, 0, 0), groups=groups)


def conv_1x1x1(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (1, 1, 1), (1, 1, 1), (0, 0, 0), groups=groups)


def conv_3x3x3(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=groups)


def conv_5x5x5(inp, oup, groups=1):
    return nn.Conv3d(inp, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)


class conv_7x7x7(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, channels, oup, groups):
        super().__init__()

        self.l2 = nn.Conv3d(channels, oup, (5, 5, 5), (1, 1, 1), (2, 2, 2), groups=groups)


    def forward(self, input):

        out = self.l2(input)

        return out

def bn_3d(dim):
    return nn.BatchNorm3d(dim)
# （1）通道注意力机制
class vchannel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        bottleneck = channels//2
        self.seavg = nn.Sequential(
            # 全局平均池化压缩为1个数
            nn.AdaptiveAvgPool3d((1,1,1)),
            nn.Conv3d(channels, bottleneck, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(bottleneck, channels, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        )

        self.semax = nn.Sequential(
            # 全局平均池化压缩为1个数
            nn.AdaptiveMaxPool3d((1,1,1)),
            nn.Conv3d(channels, bottleneck, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.GELU(),
            nn.Conv3d(bottleneck, channels, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # 获得权重矩阵
        x1 = self.seavg(input)
        x2 = self.semax(input)
        out = x1 + x2
        out = self.sigmoid(out) * input

        return out

# （2）时空注意力机制
class vl_attention(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(vl_attention, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2

        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):

        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)


        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)

        x = torch.cat([x_maxpool, x_avgpool], dim=1)


        x = self.conv(x)

        x = self.sigmoid(x)

        outputs = inputs * x

        return outputs
# （3）空间注意力机制
class t_attion(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(t_attion, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

        self.avg = nn.AdaptiveAvgPool3d((1, None, None))
        self.max = nn.AdaptiveMaxPool3d((1, None, None))

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        x_maxpool = self.max(x_maxpool)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        x_avgpool = self.avg(x_avgpool)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs

#（4）时间注意力；先池化然后conv2->1
class t2_attion(nn.Module):
    # 初始化，卷积核大小为7*7
    def __init__(self, kernel_size=7):
        # 继承父类初始化方法
        super(t2_attion, self).__init__()

        # 为了保持卷积前后的特征图shape相同，卷积时需要padding
        padding = kernel_size // 2
        # 7*7卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        self.conv = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)
        self.avgpool = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.maxpool = nn.AdaptiveMaxPool3d((None, 1, 1))
        # sigmoid函数
        self.sigmoid = nn.Sigmoid()

    # 前向传播
    def forward(self, inputs):
        # 在通道维度上最大池化 [b,1,h,w]  keepdim保留原有深度
        # 返回值是在某维度的最大值和对应的索引
        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)
        x_maxpool = self.maxpool(x_maxpool)

        # 在通道维度上平均池化 [b,1,h,w]
        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)
        x_avgpool = self.maxpool(x_avgpool)
        # 池化后的结果在通道维度上堆叠 [b,2,h,w]
        x = torch.cat([x_maxpool, x_avgpool], dim=1)

        # 卷积融合通道信息 [b,2,h,w]==>[b,1,h,w]
        x = self.conv(x)
        # 空间权重归一化
        x = self.sigmoid(x)
        # 输入特征图和空间权重相乘
        outputs = inputs * x

        return outputs
#（4）
class t1_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        bottleneck = channels//4
        self.seavg = nn.Sequential(
            # 全局平均池化压缩为1个数
            nn.AdaptiveAvgPool3d((None,1,1)),
            nn.Conv3d(channels, bottleneck, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv3d(bottleneck, 1, kernel_size=1, padding=0),
        )

        self.semax = nn.Sequential(
            # 全局平均池化压缩为1个数
            nn.AdaptiveMaxPool3d((None,1,1)),
            nn.Conv3d(channels, bottleneck, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv3d(bottleneck, 1, kernel_size=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        # 获得权重矩阵
        x1 = self.seavg(input)
        x2 = self.semax(input)
        out = x1 + x2
        out = self.sigmoid(out) * input
        return out

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class CMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1x1x1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1x1x1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.fc3 = conv_1x1x1(out_features*2, out_features)
        self.se = vchannel_attention(out_features)
        self.tk = vl_attention()
        self.tatt = t2_attion()
        self.kongjian = t_attion()
        self.act1 = act_layer()

        self.layer10 = nn.Conv3d(in_features, hidden_features, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.layer11 = nn.Conv3d(hidden_features, out_features, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.layer12 = nn.Conv3d(out_features, hidden_features, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.layer13 = nn.Conv3d(hidden_features, out_features, kernel_size=(3, 3, 3), padding=(1, 1, 1))


        self.layer10 = nn.Sequential(
            nn.Conv3d(in_features, in_features, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_features),
            nn.GELU(),
            nn.Conv3d(in_features, in_features, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_features)
        )

        self.layer11 = nn.Sequential(
            nn.Conv3d(in_features, in_features, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_features),
            nn.GELU(),
            nn.Conv3d(in_features, in_features, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.BatchNorm3d(in_features)
        )

    def forward(self, x):
        #x = torch.cat([self.se(x), self.tk(x), self.tatt(x), self.kongjian(x)], dim=1)

        #B, C, T, H, W = x.shape
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)
        #print(self.se(x).size())
        '''x = torch.cat([self.se(x)*self.tk(x),self.se(x)*self.tatt(x)*self.kongjian(x)],dim=1)
        x = self.act1(x)
        x = self.drop(x)
        x = self.fc3(x)'''
        ###x = self.drop(x)
        '''x1 = self.act1(self.layer10(x) + x)
        x = self.act1(self.layer11(x1) + x1)'''

        '''x1 = self.layer10(x)
        x1 = self.act(x1)
        x1 = self.drop(x1)

        x1 = self.layer11(x1)
        x1 = self.act(x1)
        x1 = self.drop(x1)

        x = self.layer12(x1 + x)
        x = self.act(x)
        x = self.drop(x)

        x = self.layer13(x)
        #x = self.act(x)
        x = self.drop(x)'''


        '''x = torch.cat([x, self.se(x), self.tk(x), self.tatt(x), self.kongjian(x)], dim=1)

        #x = self.act(x)
        x = x.flatten(2).transpose(1, 2)
        #print(x.size())
        x = self.fc3(x)
        #print(x.size())
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        x = self.drop(x)'''
        return x


class CBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = bn_3d(dim)
        self.norm11 = bn_3d(dim)
        self.conv1 = conv_3x3x3(dim, dim, 1)
        self.conv11 = conv_1x1x1(dim, dim, 1)

        self.conv2 = conv_5x5x5(dim, dim, 1)
        self.attn = conv_7x7x7(dim, dim, groups=dim)
        self.attn1 = conv_5x5x5(dim, dim, groups=dim)
        self.attn2 = conv_5x5x5(dim, dim, groups=dim)
        self.attn3 = conv_5x5x5(dim, dim, groups=dim)
        self.attn4 = conv_5x5x5(dim, dim, groups=dim)
        self.attn5 = conv_5x5x5(dim, dim, groups=dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_3d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.se = vchannel_attention(dim)
        self.tk = vl_attention()
        self.tatt = t2_attion()
        self.kongjian = t_attion()
        self.avgpool = nn.AdaptiveAvgPool3d((1, None, None))
        self.maxpool = nn.AdaptiveMaxPool3d((1, None, None))


        self.l1 = nn.Conv3d(dim, dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=1)
        self.nor = nn.BatchNorm3d(dim)
        self.l11 = nn.Conv3d(dim, dim, (7, 7, 7), (1, 1, 1), (3, 3, 3), groups=1)
        self.nor1 = nn.BatchNorm3d(dim)
        self.l111 = nn.Conv3d(dim, dim, (3, 3, 3), (1, 1, 1), (1, 1, 1), groups=1)
        self.nor11 = nn.BatchNorm3d(dim)
        self.l1111 = nn.Conv3d(dim, dim, (7, 7, 7), (1, 1, 1), (3, 3, 3), groups=1)


    def forward(self, x):
        x = x + self.pos_embed(x)

        x1 = x + self.drop_path(self.tk(self.se(x)))
        x1 = self.conv1(self.norm1(x1))
        out = self.nor(self.l1(x1))
        # print(out.size())
        out = self.nor1(self.l11(out))
        out = self.drop_path(self.nor11(self.l111(out)))


        x = x + self.drop_path(self.conv2(self.attn(out)))


        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class SABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2).reshape(B, C, T, H, W)
        return x


class SplitSABlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = conv_3x3x3(dim, dim, groups=dim)
        self.t_norm = norm_layer(dim)
        self.t_attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        B, C, T, H, W = x.shape
        attn = x.view(B, C, T, H * W).permute(0, 3, 2, 1).contiguous()
        attn = attn.view(B * H * W, T, C)
        attn = attn + self.drop_path(self.t_attn(self.t_norm(attn)))
        attn = attn.view(B, H * W, T, C).permute(0, 2, 1, 3).contiguous()
        attn = attn.view(B * T, H * W, C)
        residual = x.view(B, C, T, H * W).permute(0, 2, 3, 1).contiguous()
        residual = residual.view(B * T, H * W, C)
        attn = residual + self.drop_path(self.attn(self.norm1(attn)))
        attn = attn.view(B, T * H * W, C)
        out = attn + self.drop_path(self.mlp(self.norm2(attn)))
        out = out.transpose(1, 2).reshape(B, C, T, H, W)
        return out


class SpeicalPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = conv_3xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, std=False):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        if std:
            self.proj = conv_3xnxn_std(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])
        else:
            self.proj = conv_1xnxn(in_chans, embed_dim, kernel_size=patch_size[0], stride=patch_size[0])

    def forward(self, x):
        B, C, T, H, W = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, T, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, T, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        return x


class Videobranch(nn.Module):
    def __init__(self, depth=[5, 8, 20, 7], num_classes=8, img_size=224, in_chans=3, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0.3, attn_drop_rate=0., drop_path_rate=0., norm_layer=None, split=False, std=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed1 = SpeicalPatchEmbed(
            img_size=img_size, patch_size=4, in_chans=in_chans, embed_dim=embed_dim[0])

        self.patch_embed10 = PatchEmbed(
            img_size=img_size, patch_size=2, in_chans=32, embed_dim=32)

        self.patch_embed11 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=32, embed_dim=embed_dim[0], std=std)

        self.patch_embed2 = PatchEmbed(
            img_size=img_size // 4, patch_size=2, in_chans=embed_dim[0], embed_dim=embed_dim[1], std=std)



        self.patch_embed3 = PatchEmbed(
            img_size=img_size // 8, patch_size=2, in_chans=embed_dim[1], embed_dim=embed_dim[2], std=std)
        self.patch_embed4 = PatchEmbed(
            img_size=img_size // 16, patch_size=2, in_chans=embed_dim[2], embed_dim=embed_dim[3], std=std)

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depth))]  # stochastic depth decay rule
        num_heads = [dim // head_dim for dim in embed_dim]
        self.blocks1 = nn.ModuleList([
            CBlock(
                dim=embed_dim[0], num_heads=num_heads[0], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            CBlock(
                dim=embed_dim[1], num_heads=num_heads[1], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])

        self.blocks3 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[2], num_heads=num_heads[2], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0] + depth[1]],
                    norm_layer=norm_layer)
                for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
                SABlock(
                    dim=embed_dim[3], num_heads=num_heads[3], mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                    drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i + depth[0] + depth[1] + depth[2]],
                    norm_layer=norm_layer)
                for i in range(depth[3])])
        self.norm = bn_3d(embed_dim[-1])#-1

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = nn.Linear(embed_dim[-1], num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed1(x)
        x = self.pos_drop(x)

        y = []
        for blk in self.blocks1:
            x = blk(x)
            y.append(x)
        x = self.patch_embed2(x)
        for blk in self.blocks2:
            x = blk(x)
            y.append(x)
        x = self.patch_embed3(x)
        for blk in self.blocks3:
            x = blk(x)
            y.append(x)
        x = self.patch_embed4(x)
        for blk in self.blocks4:
            x = blk(x)
            y.append(x)
        x = self.norm(x)
        x = self.pre_logits(x)
        return x, y

    def forward(self, x):
        x, y = self.forward_features(x)

        x = x.flatten(2)

        return x, y


def videobranch():
    return Videobranch(
        depth=[3, 4, 8, 3], embed_dim=[64, 128, 320, 512],
        head_dim=64, drop_rate=0.1)

def conv111(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv1d(inp, oup, kernel_size, stride, 0, groups=groups)


def conv1_std(inp, oup, kernel_size=3, stride=3, groups=1):
    return nn.Conv1d(inp, oup, kernel_size, stride, 0, groups=groups)


def conv_1(inp, oup, groups=1):
    return nn.Conv1d(inp, oup, 1, 1, 0, groups=groups)


# without bn version
class conv_sp(nn.Module):
    def __init__(self, inp, oup):
        super().__init__()

        self.atrous_block1 = nn.Conv1d(inp, oup, 3, 1, padding=1, dilation=1)
        self.atrous_block6 = nn.Conv1d(inp, oup, 3, 1, padding=3, dilation=3)
        self.atrous_block12 = nn.Conv1d(inp, oup, 3, 1, padding=5, dilation=5)
        self.atrous_block18 = nn.Conv1d(inp, oup, 3, 1, padding=7, dilation=7)
        self.conv_1x1_output = nn.Conv1d(oup * 4, oup, 7, 1,padding=3)
        self.bn = nn.BatchNorm1d(oup)
        self.re = nn.GELU()
        self.bn1 = nn.BatchNorm1d(oup)
        self.re1 = nn.GELU()
        self.bn2 = nn.BatchNorm1d(oup)
        self.re2 = nn.GELU()
        self.bn3 = nn.BatchNorm1d(oup)
        self.re3 = nn.GELU()
        self.bn4 = nn.BatchNorm1d(oup)
        self.re4 = nn.GELU()
        self.bn5 = nn.BatchNorm1d(oup)
        self.re5 = nn.GELU()
        self.dropout = nn.Dropout(p=0.1)
        self.se = channel_attention(oup* 4)
        self.ba = l_attention()
        self.attn1 = conv_5(oup, oup, groups=oup)
        self.attn2 = conv_5(oup, oup, groups=oup)
        self.attn3 = conv_5(oup, oup, groups=oup)
        self.attn4 = conv_5(oup, oup, groups=oup)
        self.attn5 = conv_5(oup, oup, groups=oup)

    def forward(self, x):
        atrous_block1 = self.bn(self.atrous_block1(x))
        atrous_block6 = self.bn1(self.atrous_block6(x))
        atrous_block12 = self.bn2(self.atrous_block12(x))
        atrous_block18 = self.bn3(self.atrous_block18(x))

        net = x + self.conv_1x1_output(torch.cat([atrous_block1, atrous_block6,
                                                  atrous_block12, atrous_block18], dim=1))

        return net

def pos_embed1(inp, oup, groups=1):
    return nn.Conv1d(inp, oup, 3, 1, 1, groups=groups)


def conv_5(inp, oup, groups=1):
    return nn.Conv1d(inp, oup, 5, 1, 2, groups=groups)


def bn_1d(dim):
    return nn.BatchNorm1d(dim)

class crossavAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, q,k):

        B, N, C = q.shape


        q1 = self.q(q).reshape(B, N, 1,self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q = q1[0]
        kv = self.k(k).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)

        k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class crossvaAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(1, 2)
        return x

class crossvaAttention1(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

class CMlp1d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = conv_1(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Mlp1d(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class PatchEmbedfusion(nn.Module):


    def __init__(self, img_size=360, patch_size=16, in_chans=3, embed_dim=512, std=False):
        super().__init__()
        #img_size = to_2tuple(img_size)
        #patch_size = to_2tuple(patch_size)
        num_patches = img_size // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.act_layer = nn.GELU()
        if std:
            self.proj = conv1_std(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = conv111(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)


    def forward(self, x):

        x = self.proj(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()
        x = self.act_layer(x)

        return x

class PatchEmbedfusionx2(nn.Module):

    def __init__(self, img_size=360, patch_size=16, in_chans=3, embed_dim=512, std=False):
        super().__init__()
        #img_size = to_2tuple(img_size)
        #patch_size = to_2tuple(patch_size)
        num_patches = img_size // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.act_layer = nn.GELU()
        if std:
            self.proj = conv1_std(in_chans, embed_dim, kernel_size=7, stride=2)
        else:
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=7, stride=1, padding=3, groups=1)

    def forward(self, x):
        x = self.proj(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()
        x = self.act_layer(x)
        return x

class PatchEmbedfusionx4(nn.Module):

    def __init__(self, img_size=360, patch_size=16, in_chans=3, embed_dim=512, std=False):
        super().__init__()

        num_patches = img_size // patch_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.norm = nn.LayerNorm(embed_dim)
        self.act_layer = nn.GELU()
        if std:
            self.proj = conv1_std(in_chans, embed_dim, kernel_size=7, stride=2)
        else:
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3, groups=1)

    def forward(self, x):
        B, C, L = x.shape
        # FIXME look at relaxing size constraints

        x = self.proj(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()

        return x



class fuCBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = pos_embed1(dim, dim, groups=dim)
        self.pos_embed2 = pos_embed1(dim, dim, groups=dim)
        self.pos_embed3 = pos_embed1(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.norms = norm_layer(dim)
        self.norm12 = norm_layer(dim)
        self.norm13 = norm_layer(dim)
        self.norm14 = norm_layer(dim)
        self.conv1 = conv_1(dim, dim, 1)
        self.conv2 = conv_1(dim*2, dim, 1)
        self.conv3 = conv_1(dim, dim, 1) #MultiheadAttention(d_model, n_head)
        #self.attn = conv_5(dim, dim, groups=dim)
        self.attn = crossavAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.attn2 = crossavAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.attn3 = crossvaAttention1(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp1d(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.re = nn.GELU()
        self.mattn = MultiheadAttention(dim, num_heads)

    def attention(self, x,y):
        attn_mask =  None
        return self.mattn(x, y, y, need_weights=False, attn_mask=attn_mask)[0]
    def attention1(self, x):
        attn_mask =  None
        return self.mattn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x,v):

        sum = self.re(x + v)
        sum = sum.transpose(1, 2)



        x = x + self.pos_embed(x)
        x = x.transpose(1,2)
        v = v + self.pos_embed2(v)
        v = v.transpose(1, 2)
        av = self.attention(self.norm1(sum), self.norm12(x))
        va = self.attention(self.norm14(sum), self.norm13(v))

        sum1 = sum.transpose(1, 2)
        sum1 = sum1 + self.pos_embed3(sum1)
        sum1 = sum1.transpose(1, 2)
        sum1 = self.attention1(self.norms(sum1))



        x = av + va + sum + sum1
        x = av + va + sum + sum1 + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.transpose(1, 2)
        return x

class fuCBlock2(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = pos_embed1(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.conv1 = conv_1(dim, dim, 1)
        self.conv2 = conv_1(dim, dim, 1)
        #self.attn = conv_5(dim, dim, groups=dim)
        self.attn = crossvaAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp1d(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x1 = x.transpose(1,2)


        x = x + self.drop_path(self.conv2(self.attn(self.norm1(x1))))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class fuCBlock3(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = pos_embed1(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = crossvaAttention1(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        #self.attn = Attention4D(dim, act_layer=act_layer, stride=1)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp1d(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mattn = MultiheadAttention(dim, num_heads)

    def attention1(self, x):
        attn_mask =  None
        return self.mattn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x):
        x = x + self.pos_embed(x)

        x = x.transpose(1, 2)
        x = x + self.drop_path(self.attention1(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #x = x + self.attn(self.norm1(x))
        #x = x + self.mlp(self.norm2(x))
        x = x.transpose(1, 2)
        return x

class fuCBlock31(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = pos_embed1(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = crossvaAttention1(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        #self.attn = Attention4D(dim, act_layer=act_layer, stride=1)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp1d(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        #x = x + self.pos_embed(x)

        x = x.transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        #x = x + self.attn(self.norm1(x))
        #x = x + self.mlp(self.norm2(x))
        x = x.transpose(1, 2)
        return x

class fusion(nn.Module):
    def __init__(self):
        super().__init__()


        self.block_fusion1 = nn.Sequential(
            #nn.ReflectionPad1d(6),
            nn.Conv1d(512, 512, 7, 4, 3),
            nn.BatchNorm1d(512),


        )

        self.block_fusion10 = nn.Sequential(
            # nn.ReflectionPad1d(6),
            nn.Conv1d(512, 512, 7, 1, 3),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 512, 7, 1, 3),
            nn.BatchNorm1d(512),

        )



        self.block_fusion2 = nn.Sequential(
            # nn.ReflectionPad1d(6),
            nn.Conv1d(512, 512, 7, 4, 3),
            nn.BatchNorm1d(512)

        )

        self.block_fusion20 = nn.Sequential(
            # nn.ReflectionPad1d(6),
            nn.Conv1d(512, 512, 7, 1, 3),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 512, 7, 1, 3),
            nn.BatchNorm1d(512),

        )




        self.mel1 = nn.Sequential(
            # nn.ReflectionPad1d(6),
            nn.Conv1d(1024, 1024, 7, 1, 3),
            nn.BatchNorm1d(1024)
        )
        self.re = nn.GELU()
        self.re1 = nn.GELU()
        self.re2 = nn.GELU()
        self.dropout = nn.Dropout(p=0.4)
        self.pllot = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(11264, 8)
        self.fc11 = nn.Linear(5632, 8)
        self.fc111 = nn.Linear(512, 8)




    def forward(self, x):

        out = self.re(x)


        out = self.block_fusion1(out)


        out = self.re1(out + self.block_fusion10(self.re(out)))



        out = self.block_fusion2(out)

        out = out + self.block_fusion20(self.re(out))

        out = self.pllot(out)

        out = out.flatten(1, 2)
        out = self.dropout(out)

        out = self.fc111(out)



        return out


class fusonform(nn.Module):
    def __init__(self, depth=[5, 8, 20, 7], num_classes=8, img_size=224, in_chans=3, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., norm_layer=None, split=False, std=False):
        super().__init__()
        std = False
        depth = [1, 2, 4, 3, 1, 2, 3]#3, 4, 8, 3     1, 2, 4, 3
        self.patch_embed2 = PatchEmbedfusionx2(
            img_size=300 // 2, patch_size=2, in_chans=512, embed_dim=512, std=std)
        self.patch_embed21 = PatchEmbedfusionx2(
            img_size=300 // 2, patch_size=2, in_chans=512, embed_dim=512, std=std)
        self.patch_embed3 = PatchEmbedfusionx4(
            img_size=150 // 4, patch_size=4, in_chans=512, embed_dim=512, std=std)
        self.patch_embed4 = PatchEmbedfusionx4(
            img_size=75 // 4, patch_size=4, in_chans=512, embed_dim=512, std=std)
        self.patch_embed5 = PatchEmbedfusion(
            img_size=45 // 4, patch_size=2, in_chans=512, embed_dim=512, std=std)
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depth))]
        self.pre_logits = nn.Identity()

        self.down = fusion()



        self.cbn = fuCBlock(
            dim=512, num_heads=512 // 64, mlp_ratio=4, qkv_bias=True, qk_scale=None,
            drop=0.3, attn_drop=0, drop_path=dpr[0], norm_layer=norm_layer)

        self.re = nn.GELU()
        self.re1 = nn.GELU()

        self.blocks1 = nn.ModuleList([
            fuCBlock(
                dim=512, num_heads=512 // 64, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[i + depth[0] + depth[1]+ depth[2]], norm_layer=norm_layer)
            for i in range(depth[4])])
        self.blocks2 = nn.ModuleList([
            fuCBlock3(
                dim=512, num_heads=512 // 64, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[i + depth[0] + depth[1]+ depth[2] + depth[4]], norm_layer=norm_layer)
            for i in range(depth[5])])
        self.blocks3 = nn.ModuleList([
            fuCBlock3(
                dim=512, num_heads=512 // 64, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[i + depth[0] + depth[1] + depth[2] + depth[3] + depth[4] + depth[5]], norm_layer=norm_layer)
            for i in range(depth[6])])
        self.blocks4 = nn.ModuleList([
            fuCBlock3(
                dim=512, num_heads=512 // 64, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[0], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.norm = bn_1d(512)  # -1
        #self.head = nn.Linear(9216, 2400)
        #self.head = nn.Linear(2560, 8)
        self.head = nn.Linear(18432, 4800)
        self.head1 = nn.Linear(512, 8)
        self.downfinal = nn.Conv1d(512, 512, kernel_size=7, stride=4, padding=3)
        self.dropout = nn.Dropout(p=0.4)
        self.dropout1 = nn.Dropout(p=0.4)
        self.pllot = nn.AdaptiveAvgPool1d(1)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


    def forward(self, v, a, mel):
        x = a + mel
        all = a + mel + v



        for blk in self.blocks1:
            x = blk(x,v)


        x = self.norm(x)

        x = self.down(x+all)

        return x
class SpeicalPatchEmbeda4(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=512, std=False):
        super().__init__()
        #img_size = to_2tuple(img_size)
        #patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size

        self.norm = nn.LayerNorm(embed_dim)
        self.act_layer = nn.GELU()
        if std:
            self.proj = conv1_std(in_chans, embed_dim, kernel_size=7, stride=2)
        else:
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=15, stride=4, padding=7)

        self.proj1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, stride=1, padding=3, groups=1)
        self.norm1 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, L = x.shape
        # FIXME look at relaxing size constraints
        # assert H == self.img_size[0] and W == self.img_size[1], \
        #     f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()
        x = self.act_layer(x)
        x = self.proj1(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()
        return x
class SpeicalPatchEmbedax4(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=16, in_chans=3, embed_dim=512, std=False):
        super().__init__()
        #img_size = to_2tuple(img_size)
        #patch_size = to_2tuple(patch_size)

        self.patch_size = patch_size

        self.norm = nn.LayerNorm(embed_dim)
        self.act_layer = nn.GELU()
        self.act_layer1 = nn.GELU()
        if std:
            self.proj = conv1_std(in_chans, embed_dim, kernel_size=7, stride=2)
        else:
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=7, stride=4, padding=3)

        self.proj1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, stride=1, padding=3, groups=1)
        self.norm1 = nn.LayerNorm(embed_dim)



    def forward(self, x):

        x = self.proj(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()
        x = self.act_layer(x)
        x = self.proj1(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()

        return x



class channel_attention(nn.Module):
    # 初始化, in_channel代表输入特征图的通道数, ratio代表第一个全连接的通道下降倍数
    def __init__(self, channels, bottleneck=128):
        super().__init__()
        bottleneck = channels
        self.seavg = nn.Sequential(
            # 全局平均池化压缩为1个数
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
        )

        self.semax = nn.Sequential(
            # 全局平均池化压缩为1个数
            nn.AdaptiveMaxPool1d(1),
            nn.Conv1d(channels, bottleneck, kernel_size=1, padding=0),
            nn.GELU(),
            nn.Conv1d(bottleneck, channels, kernel_size=1, padding=0),
        )
        self.sigmoid = nn.Sigmoid()
        self.re = nn.GELU()

    def forward(self, input):
        # 获得权重矩阵
        x1 = self.seavg(input)
        x2 = self.semax(input)
        out = x1 + x2
        #out = self.sigmoid(out) * input
        out = self.re(out * input)
        return out


class l_attention(nn.Module):

    def __init__(self, kernel_size=7):

        super(l_attention, self).__init__()


        padding = kernel_size // 2

        self.conv = nn.Conv1d(in_channels=2, out_channels=1, kernel_size=kernel_size,
                              padding=padding, bias=False)

        self.sigmoid = nn.Sigmoid()
        self.re = nn.GELU()


    def forward(self, inputs):

        x_maxpool, _ = torch.max(inputs, dim=1, keepdim=True)


        x_avgpool = torch.mean(inputs, dim=1, keepdim=True)

        x = torch.cat([x_maxpool, x_avgpool], dim=1)


        x = self.conv(x)

        outputs = self.re(inputs * x)

        return outputs

class CMlpres(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = conv_1(in_features, hidden_features)
        self.act = act_layer()
        self.act1 = act_layer()
        self.fc2 = conv_1(hidden_features, out_features)
        self.fc3 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.latt = l_attention()
        self.cha = channel_attention(out_features)
        self.norm1 = bn_1d(out_features)
        self.norm2 = bn_1d(hidden_features)

    def forward(self, x):



        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        #x = self.cha(x)*self.latt(x)
        #x = self.act1(x)
        x = self.drop(x)

        return x

class Mlp1da(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.latt = l_attention()
        self.cha = channel_attention(hidden_features)

    def forward(self, x):

        x = self.fc1(x)

        x = self.act(x)
        x = self.drop(x)

        x = self.fc2(x)
        x = self.drop(x)

        return x

class fuCBlock333(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = pos_embed1(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = crossvaAttention1(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp1da(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.mattn = MultiheadAttention(dim, num_heads)

    def attention1(self, x):
        attn_mask =  None
        return self.mattn(x, x, x, need_weights=False, attn_mask=attn_mask)[0]

    def forward(self, x):
        x = x + self.pos_embed(x)

        x = x.transpose(1, 2)
        x = x + self.drop_path(self.attention1(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        x = x.transpose(1, 2)
        return x

class aCBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = pos_embed1(dim, dim, groups=dim)
        self.norm1 = bn_1d(dim)
        self.norm3 = bn_1d(dim)
        self.conv1 = conv_sp(dim, dim)
        self.conv2 = conv_1(dim, dim, 1)
        self.conv3 = conv_1(dim, dim, 1)
        self.conv4 = conv_1(dim, dim)
        self.attn = conv_5(dim, dim, groups=dim)
        self.attn2 = conv_5(dim, dim, groups=dim)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlpres(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.latt = l_attention()
        self.cha = channel_attention(dim)

    def forward(self, x):
        x = x + self.pos_embed(x)


        x1 =x + self.conv2(self.attn(self.conv1(self.norm1(x))))
        x = x + self.drop_path(self.conv3(self.attn2(self.conv4(self.norm3(x1)))))

        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
class afuCBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.pos_embed = pos_embed1(dim, dim, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = crossvaAttention1(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = bn_1d(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = CMlp1d(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.pos_embed(x)

        x = x.transpose(1, 2)
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x.transpose(1, 2)
        x = x + self.drop_path(self.mlp(self.norm2(x)))


        return x


class audiobranch(nn.Module):
    def __init__(self, depth=[5, 8, 20, 7], num_classes=8, img_size=224, in_chans=3, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., norm_layer=None, split=False, std=False):
        super().__init__()
        std = False
        depth = [1, 2, 4, 3]

        self.patch_embed1 = SpeicalPatchEmbeda4(
            patch_size=4, in_chans=2, embed_dim=32)
        self.patch_embed2 = SpeicalPatchEmbedax4(
            in_chans=32, embed_dim=64, std=std)
        self.patch_embed3 = SpeicalPatchEmbedax4(
            in_chans=64, embed_dim=128, std=std)
        self.patch_embed4 = SpeicalPatchEmbedax4(
            in_chans=128, embed_dim=512, std=std)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depth))]

        self.blocks1 = nn.ModuleList([
            aCBlock(
                dim=32, num_heads=64 // 64, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks22 = nn.ModuleList([
            afuCBlock(
                dim=64, num_heads=2, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[i + depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])

        self.blocks2 = nn.ModuleList([
            aCBlock(
                dim=64, num_heads=2, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[i + depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])

        self.blocks3 = nn.ModuleList([
            fuCBlock333(
                dim=128, num_heads=4, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[i + depth[0] + depth[1]], norm_layer=norm_layer)
            for i in range(depth[2])])
        self.blocks4 = nn.ModuleList([
            fuCBlock333(
                dim=512, num_heads=512 // 64, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[i + depth[0] + depth[1] + depth[2]], norm_layer=norm_layer)
            for i in range(depth[3])])


        self.re = nn.GELU()
        self.norm = nn.BatchNorm1d(512)

        for name, p in self.named_parameters():
            # fill proj weight with 1 here to improve training dynamics. Otherwise temporal attention inputs
            # are multiplied by 0*0, which is hard for the model to move out of.
            if 't_attn.qkv.weight' in name:
                nn.init.constant_(p, 0)
            if 't_attn.qkv.bias' in name:
                nn.init.constant_(p, 0)
            if 't_attn.proj.weight' in name:
                nn.init.constant_(p, 1)
            if 't_attn.proj.bias' in name:
                nn.init.constant_(p, 0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed1(x)
        x1 = x
        for blk in self.blocks1:
            x = x + blk(x)
        x = self.patch_embed2(x+x1)
        x2 = x
        for blk in self.blocks2:
            x = x + blk(x)
        x = self.patch_embed3(x+x2)
        x3 = x
        for blk in self.blocks3:
            x = x + blk(x)
        x = self.patch_embed4(x+x3)
        x4 = x
        for blk in self.blocks4:
            x = x4 + blk(x)

        x = self.norm(x)

        return x

class amel1(nn.Module):


    def __init__(self, patch_size=16, in_chans=3, embed_dim=512, std=False):
        super().__init__()


        self.patch_size = patch_size

        self.norm = nn.LayerNorm(embed_dim)
        self.act_layer = nn.GELU()
        self.act_layer1 = nn.GELU()
        if std:
            self.proj = conv1_std(in_chans, embed_dim, kernel_size=7, stride=2)
        else:
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=7, stride=1, padding=3)

        self.proj1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, stride=1, padding=3, groups=1)
        self.norm1 = nn.LayerNorm(embed_dim)



    def forward(self, x):


        x = self.proj(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()
        x = self.act_layer(x)
        x = self.proj1(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()

        return x

class amel2(nn.Module):


    def __init__(self, patch_size=16, in_chans=3, embed_dim=512, std=False):
        super().__init__()

        self.patch_size = patch_size

        self.norm = nn.LayerNorm(embed_dim)
        self.act_layer = nn.GELU()
        self.act_layer1 = nn.GELU()
        if std:
            self.proj = conv1_std(in_chans, embed_dim, kernel_size=7, stride=2)
        else:
            self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=7, stride=1, padding=3)

        self.proj1 = nn.Conv1d(embed_dim, embed_dim, kernel_size=7, stride=1, padding=3, groups=1)
        self.norm1 = nn.LayerNorm(embed_dim)



    def forward(self, x):

        x = self.proj(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()
        x = self.act_layer(x)
        x = self.proj1(x)
        B, C, L = x.shape
        x = x.transpose(1, 2)
        x = self.norm1(x)
        x = x.reshape(B, L, -1).permute(0, 2, 1).contiguous()

        return x

class amel(nn.Module):
    def __init__(self, depth=[5, 8, 20, 7], num_classes=8, img_size=224, in_chans=3, embed_dim=[64, 128, 320, 512],
                 head_dim=64, mlp_ratio=4., qkv_bias=True, qk_scale=None, representation_size=None,
                 drop_rate=0.1, attn_drop_rate=0., drop_path_rate=0., norm_layer=None, split=False, std=False):
        super().__init__()
        std = False
        depth = [4, 8]

        self.patch_embed1 = nn.Sequential(
            nn.ReflectionPad1d((1,0)),
            amel1(in_chans=360, embed_dim=512, std=std))

        self.patch_embed2 = amel2(
            in_chans=512, embed_dim=512, std=std)

        self.patch_embed3 = amel2(
            in_chans=512, embed_dim=512, std=std)

        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depth))]
        self.blocks1 = nn.ModuleList([
            aCBlock(
                dim=512, num_heads=512 // 64, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])
        self.blocks2 = nn.ModuleList([
            fuCBlock333(
                dim=512, num_heads=512 // 64, mlp_ratio=4, qkv_bias=True, qk_scale=None,
                drop=0.1, attn_drop=0, drop_path=dpr[i+depth[0]], norm_layer=norm_layer)
            for i in range(depth[1])])


        self.block_fusion1 = nn.Sequential(
            #nn.ReflectionPad1d(1),
            nn.Conv1d(320, 512, 7, 1, 2),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 512, 7, 1, 3),
            nn.BatchNorm1d(512),
            nn.GELU(),

        )



        self.block_fusion2 = nn.Sequential(
            # nn.ReflectionPad1d(6),
            nn.Conv1d(512, 512, 7, 1, 3),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Conv1d(512, 512, 7, 1, 3),
            nn.BatchNorm1d(512)

        )
        self.norm = nn.BatchNorm1d(512)







    def forward(self, mel):


        x = self.patch_embed1(mel)
        for blk in self.blocks1:
            x = x + blk(x)

        x = self.patch_embed2(x)
        x2 = x
        for blk in self.blocks2:
            x = x2 + blk(x)



        out = self.norm(x)



        return out







class fusoinav(nn.Module):
    def __init__(self):
        super().__init__()
        self.block_downv = videobranch()

        self.block_downa = audiobranch()

        self.mel = amel()


        self.fu = fusonform()

    def forward(self, v, a, mel):
        xv, y= self.block_downv(v)
        xa = self.block_downa(a)
        xmel = self.mel(mel)
        # print('xv')
        # print(xv)



        out = self.fu(xv, xa, xmel)


        return out

if __name__ == "__main__":
    print('EAVFomer')