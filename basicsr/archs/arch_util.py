import math
import torch
from torch import nn as nn
from torch.nn import functional as F
from torch.nn import init as init
from torch.nn.modules.batchnorm import _BatchNorm
from basicsr.archs.kpn_pixel import IDynamicDWConv
from basicsr.utils import get_root_logger
import datetime

@torch.no_grad()
def default_init_weights(module_list, scale=1, bias_fill=0, **kwargs):
    """Initialize network weights.

    Args:
        module_list (list[nn.Module] | nn.Module): Modules to be initialized.
        scale (float): Scale initialized weights, especially for residual
            blocks. Default: 1.
        bias_fill (float): The value to fill bias. Default: 0
        kwargs (dict): Other arguments for initialization function.
    """
    if not isinstance(module_list, list):
        module_list = [module_list]
    for module in module_list:
        for m in module.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight, **kwargs)
                m.weight.data *= scale
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)
            elif isinstance(m, _BatchNorm):
                init.constant_(m.weight, 1)
                if m.bias is not None:
                    m.bias.data.fill_(bias_fill)


def make_layer(basic_block, num_basic_block, **kwarg):
    """Make layers by stacking the same blocks.

    Args:
        basic_block (nn.module): nn.module class for basic block.
        num_basic_block (int): number of blocks.

    Returns:
        nn.Sequential: Stacked blocks in nn.Sequential.
    """
    layers = []
    for _ in range(num_basic_block):
        layers.append(basic_block(**kwarg))
    return nn.Sequential(*layers)


class ResidualBlockNoBN3D(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN3D, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class ResidualBlockNoBN2D(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, res_scale=1, pytorch_init=False):
        super(ResidualBlockNoBN2D, self).__init__()
        self.res_scale = res_scale
        self.conv1 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.conv2 = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.relu = nn.ReLU(inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.relu(self.conv1(x)))
        return identity + out * self.res_scale


class TSM(nn.Module):
    def __init__(self, num_feat=64):
        super(TSM, self).__init__()
        self.tsm = TemporalShift(nn.Sequential(), n_div=8, inplace=False)
        self.conv = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # split segment & tsm3
        if(x.size(1) == 10):
            x_list = torch.split(x, [3, 3, 4], dim=1)
            x0 = self.tsm(x_list[0])
            x1 = self.tsm(x_list[1])
            x2 = self.tsm(x_list[2])
            x = torch.cat([x0, x1, x2], dim=1)  # b t c h w
        else:
            x_list = torch.split(x, 4, dim=1)
            if(len(x_list)==1):
                x = self.tsm(x_list[0])
            else:
                x0 = self.tsm(x_list[0])
                x1 = self.tsm(x_list[1])
                x = torch.cat([x0, x1], dim=1)  # b t c h w
        # tsm residual block
        x = x.permute(0, 2, 1, 3, 4)            # b c t h w
        identity = x
        out = self.relu(self.conv(x) + identity)
        return out.permute(0, 2, 1, 3, 4)       # b t c h w


def manual_padding_1(lrs):
    x_0 = lrs[:, 1, :, :, :].unsqueeze(1)
    x_t = lrs[:, -2, :, :, :].unsqueeze(1)
    lrs = torch.cat([x_0, lrs], dim=1)
    lrs = torch.cat([lrs, x_t], dim=1)
    return lrs

class conv2d_extractor(nn.Module):
    def __init__(self, num_feat=64):
        super(conv2d_extractor, self).__init__()
        self.conv3d_extractor = nn.Conv3d(num_feat, num_feat, (1, 3, 3), 1, (0, 1, 1), bias=True)

    def forward(self, x):
        lrs_feature = self.conv3d_extractor(x)  # b 64 t 256 256
        return lrs_feature

###############################
# RCAB
###############################
def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias)

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks):
        super(ResidualGroup, self).__init__()
        modules_body = [
            RCAB(
                conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=act, res_scale=res_scale) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class ResidualBlockNoBN(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, pytorch_init=False):
        super(ResidualBlockNoBN, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv2(self.lrelu(self.conv1(x)))
        return identity + out


class ResidualBlockNoBN_DW(nn.Module):
    """Residual block without BN.

    It has a style of:
        ---Conv-ReLU-Conv-+-
         |________________|

    Args:
        num_feat (int): Channel number of intermediate features.
            Default: 64.
        res_scale (float): Residual scale. Default: 1.
        pytorch_init (bool): If set to True, use pytorch default init,
            otherwise, use default_init_weights. Default: False.
    """

    def __init__(self, num_feat=64, pytorch_init=False):
        super(ResidualBlockNoBN_DW, self).__init__()
        self.conv1 = nn.Conv2d(num_feat, num_feat*2, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(num_feat*2, num_feat*2, 3, 1, 1, bias=True, groups=num_feat*2)
        self.conv3 = nn.Conv2d(num_feat*2, num_feat, 1, 1, 0, bias=True)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        if not pytorch_init:
            default_init_weights([self.conv1, self.conv2], 0.1)

    def forward(self, x):
        identity = x
        out = self.conv3(self.lrelu(self.conv2(self.lrelu(self.conv1(x)))))
        return identity + out


class ResidualBlockNoBN_HiLo(nn.Module):
    def __init__(self, num_feat=64):
        super(ResidualBlockNoBN_HiLo, self).__init__()
        self.num_feat = num_feat
        self.conv1 = nn.Conv2d(num_feat//2, num_feat//2, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat//2, num_feat//2, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat//2, num_feat//2, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat//2, num_feat//2, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.up = nn.ConvTranspose2d(num_feat//2, num_feat//2, 4, 2, 1)

    def forward(self, x):
        x1, x2 = torch.split(x, self.num_feat//2, dim=1)
        Hi = self.conv2(self.lrelu(self.conv1(x1)))
        Lo = F.interpolate(self.conv4(self.down2(self.conv3(self.down1(x2)))), scale_factor=4, mode='bilinear')
        y = torch.cat([Hi, Lo], dim=1)
        return y + x


class ResidualBlock2D_fft(nn.Module):
    def __init__(self, num_feat=64):
        super(ResidualBlock2D_fft, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat, num_feat, 3, 1, 1),
        )
        self.main_fft = nn.Sequential(
            nn.Conv2d(num_feat * 2, num_feat * 2, 1, 1, 0),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(num_feat * 2, num_feat * 2, 1, 1, 0),
        )
        self.norm = 'backward'

    def forward(self, x):
        _, _, H, W = x.shape
        dim = 1
        y = torch.fft.rfft2(x, norm=self.norm)
        y_imag = y.imag
        y_real = y.real
        y_f = torch.cat([y_real, y_imag], dim=dim)
        y = self.main_fft(y_f)
        y_real, y_imag = torch.chunk(y, 2, dim=dim)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.main(x) + x + y


class ResidualBlock3D_fft(nn.Module):
    def __init__(self, num_feat=64):
        super(ResidualBlock3D_fft, self).__init__()
        self.main = nn.Sequential(
            nn.Conv3d(num_feat, num_feat, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat, num_feat, 3, 1, 1),
        )
        self.main_fft = nn.Sequential(
            nn.Conv3d(num_feat * 2, num_feat * 2, (3, 1, 1), 1, (1, 0, 0)),
            nn.ReLU(inplace=True),
            nn.Conv3d(num_feat * 2, num_feat * 2, (3, 1, 1), 1, (1, 0, 0)),
        )
        self.norm = 'backward'

    def forward(self, x):
        B, T, C, H, W = x.shape
        fft_list = []
        for i in range(0, T):
            feat_now = x[:, i, :, :, :]
            y = torch.fft.rfft2(feat_now, norm=self.norm)
            y_imag = y.imag
            y_real = y.real
            y_f = torch.cat([y_real, y_imag], dim=1)
            fft_list.append(y_f)
        fft_y = torch.stack(fft_list, dim=2)
        fft_y = self.main_fft(fft_y)

        rfft_list = []
        for i in range(0, T):
            feat_now = fft_y[:, :, i, :, :]
            y_real, y_imag = torch.chunk(feat_now, 2, dim=1)
            y = torch.complex(y_real, y_imag)
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
            rfft_list.append(y)
        rfft_y = torch.stack(rfft_list, dim=2)
        return self.main(x.permute(0, 2, 1, 3, 4)) + x.permute(0, 2, 1, 3, 4) + rfft_y


class Wave_Block(nn.Module):
    def __init__(self, num_feat=64):
        super(Wave_Block, self).__init__()
        self.x11_conv1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.x11_conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=True)
        self.x12_conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x12_conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x21_conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x21_conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x22_conv1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.x22_conv2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0, bias=True)
        self.relu = nn.ReLU(inplace=True)
        default_init_weights([self.x11_conv1, self.x11_conv2, self.x12_conv1, self.x12_conv2,
                              self.x21_conv1, self.x21_conv2, self.x22_conv1, self.x22_conv2], 0.1)

    def forward(self, x):
        x11 = x[0:1, :, :, :]
        x12 = x[1:2, :, :, :]
        x21 = x[2:3, :, :, :]
        x22 = x[3:4, :, :, :]
        x11 = self.x11_conv2(self.relu(self.x11_conv1(x11))) + x11
        x12 = self.x12_conv2(self.relu(self.x12_conv1(x12)))
        x21 = self.x21_conv2(self.relu(self.x21_conv1(x21)))
        x22 = self.x22_conv2(self.relu(self.x22_conv1(x22)))
        x = torch.cat([x11, x12, x21, x22], dim=0)
        return x


class DynamicBlock(nn.Module):
    def __init__(self, num_feat=64, pytorch_init=False):
        super(DynamicBlock, self).__init__()
        self.kernel_conv_pixel1 = IDynamicDWConv(num_feat, 3, 1)
        self.pointwise1 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.kernel_conv_pixel2 = IDynamicDWConv(num_feat, 3, 1)
        self.pointwise2 = nn.Conv2d(num_feat, num_feat, 1, 1, 0)
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x):
        identity = x
        out = self.pointwise2(self.kernel_conv_pixel2(self.lrelu(self.pointwise1(self.kernel_conv_pixel1(x)))))
        return identity + out


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. ' 'Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)


def flow_warp(x, flow, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    """Warp an image or feature map with optical flow.

    Args:
        x (Tensor): Tensor with size (n, c, h, w).
        flow (Tensor): Tensor with size (n, h, w, 2), normal value.
        interp_mode (str): 'nearest' or 'bilinear'. Default: 'bilinear'.
        padding_mode (str): 'zeros' or 'border' or 'reflection'.
            Default: 'zeros'.
        align_corners (bool): Before pytorch 1.3, the default value is
            align_corners=True. After pytorch 1.3, the default value is
            align_corners=False. Here, we use the True as default.

    Returns:
        Tensor: Warped image or feature map.
    """
    assert x.size()[-2:] == flow.size()[1:3]
    b, c, h, w = x.size()
    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + flow
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    # mask
    # mask = torch.autograd.Variable(torch.ones(b, 1, h, w)).cuda()
    # mask = F.grid_sample(mask, vgrid)
    # mask[mask < 0.999] = 0
    # mask[mask > 0] = 1

    # TODO, what if align_corners=False
    return output


def resize_flow(flow, size_type, sizes, interp_mode='bilinear', align_corners=False):
    """Resize a flow according to ratio or shape.

    Args:
        flow (Tensor): Precomputed flow. shape [N, 2, H, W].
        size_type (str): 'ratio' or 'shape'.
        sizes (list[int | float]): the ratio for resizing or the final output
            shape.
            1) The order of ratio should be [ratio_h, ratio_w]. For
            downsampling, the ratio should be smaller than 1.0 (i.e., ratio
            < 1.0). For upsampling, the ratio should be larger than 1.0 (i.e.,
            ratio > 1.0).
            2) The order of output_size should be [out_h, out_w].
        interp_mode (str): The mode of interpolation for resizing.
            Default: 'bilinear'.
        align_corners (bool): Whether align corners. Default: False.

    Returns:
        Tensor: Resized flow.
    """
    _, _, flow_h, flow_w = flow.size()
    if size_type == 'ratio':
        output_h, output_w = int(flow_h * sizes[0]), int(flow_w * sizes[1])
    elif size_type == 'shape':
        output_h, output_w = sizes[0], sizes[1]
    else:
        raise ValueError(f'Size type should be ratio or shape, but got type {size_type}.')

    input_flow = flow.clone()
    ratio_h = output_h / flow_h
    ratio_w = output_w / flow_w
    input_flow[:, 0, :, :] *= ratio_w
    input_flow[:, 1, :, :] *= ratio_h
    resized_flow = F.interpolate(
        input=input_flow, size=(output_h, output_w), mode=interp_mode, align_corners=align_corners)
    return resized_flow


# TODO: may write a cpp file
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


class PixelShufflePack(nn.Module):
    """ Pixel Shuffle upsample layer.
    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        scale_factor (int): Upsample ratio.
        upsample_kernel (int): Kernel size of Conv layer to expand channels.
    Returns:
        Upsampled feature map.
    """

    def __init__(self, in_channels, out_channels, scale_factor,
                 upsample_kernel):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale_factor = scale_factor
        self.upsample_kernel = upsample_kernel
        self.upsample_conv = nn.Conv2d(
            self.in_channels,
            self.out_channels * scale_factor * scale_factor,
            self.upsample_kernel,
            padding=(self.upsample_kernel - 1) // 2)
        default_init_weights([self.upsample_conv], 1)

    def forward(self, x):
        """Forward function for PixelShufflePack.
        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
        Returns:
            Tensor: Forward results.
        """
        x = self.upsample_conv(x)
        x = F.pixel_shuffle(x, self.scale_factor)
        return x


class ResidualBlocksWithInputConv(nn.Module):
    """Residual blocks with a convolution in front.
    Args:
        in_channels (int): Number of input channels of the first conv.
        out_channels (int): Number of channels of the residual blocks.
            Default: 64.
        num_blocks (int): Number of residual blocks. Default: 30.
    """

    def __init__(self, in_channels, out_channels=64, num_blocks=30):
        super().__init__()

        main = []

        # a convolution used to match the channels of the residual blocks
        main.append(nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=True))
        main.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))

        # residual blocks
        main.append(
            make_layer(
                ResidualBlockNoBN, num_blocks, num_feat=out_channels))

        self.main = nn.Sequential(*main)

    def forward(self, feat):
        """
        Forward function for ResidualBlocksWithInputConv.
        Args:
            feat (Tensor): Input feature with shape (n, in_channels, h, w)
        Returns:
            Tensor: Output feature with shape (n, out_channels, h, w)
        """
        return self.main(feat)