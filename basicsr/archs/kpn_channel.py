import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers
import basicsr.archs.blocks as blocks
from einops import rearrange



##########################################################################
## Layer Norm

def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x,h,w):
    return rearrange(x, 'b (h w) c -> b c h w',h=h,w=w)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma+1e-5) * self.weight

class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class DynamicDWConv(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        Block1 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels)
                  for _ in range(3)]
        Block2 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels)
                  for _ in range(3)]
        self.tokernel = nn.Conv2d(channels, kernel_size ** 2 * self.channels, 1, 1, 0)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.Block1 = nn.Sequential(*Block1)
        self.Block2 = nn.Sequential(*Block2)

    def forward(self, x):
        b, c, h, w = x.shape
        weight = self.tokernel(self.pool(self.Block2(self.maxpool(self.Block1(self.avgpool(x))))))
        weight = weight.view(b * self.channels, 1, self.kernel_size, self.kernel_size)
        x = F.conv2d(x.reshape(1, -1, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.view(b, c, x.shape[-2], x.shape[-1])
        return x

class DynamicDWConv3D(nn.Module):
    def __init__(self, channels, kernel_size, stride=1, groups=1):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = kernel_size // 2
        self.groups = groups

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        Block1 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels)
                  for _ in range(3)]
        Block2 = [nn.Conv2d(channels, channels, kernel_size, 1, kernel_size//2, groups=channels)
                  for _ in range(3)]
        self.tokernel = nn.Conv2d(channels, 3 * kernel_size ** 2 * self.channels, 1, 1, 0)
        self.bias = nn.Parameter(torch.zeros(channels))
        self.Block1 = nn.Sequential(*Block1)
        self.Block2 = nn.Sequential(*Block2)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x_mean = torch.mean(x, dim=2)
        weight = self.tokernel(self.pool(self.Block2(self.maxpool(self.Block1(self.avgpool(x_mean))))))
        weight = weight.view(b * c, 1, 3, self.kernel_size, self.kernel_size)
        x = F.conv3d(x.reshape(1, b*c, t, h, w), weight, self.bias.repeat(b), stride=self.stride, padding=self.padding, groups=b * self.groups)
        x = x.squeeze(dim=0).reshape(b, c, t, h, w)
        return x


# net = DynamicDWConv3D(channels=64, kernel_size=3, stride=1, groups=64).cuda()
# x1 = torch.randn(2, 64, 10, 256, 256).cuda()
# y = net(x1)
# print(y.shape)
