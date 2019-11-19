import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import init
from torchvision import models


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal_(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

    return feat_mean, feat_std


def adain(content_feat, style_feat, soft):
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    soft_mean = style_mean * soft + content_mean * (1.0 - soft)
    soft_std = style_std * soft + content_std * (1.0 - soft)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * soft_std.expand(size) + soft_mean.expand(size)


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False, layer=None):
        super(Vgg19, self).__init__()
        self.layer = layer

        vgg_pretrained_features = models.vgg19(pretrained=True).features

        if layer == 'four':
            self.slice = nn.Sequential()
            for x in range(21):
                self.slice.add_module(str(x), vgg_pretrained_features[x])

        elif layer == 'five':
            self.slice = nn.Sequential()
            for x in range(30):
                self.slice.add_module(str(x), vgg_pretrained_features[x])

        else:
            self.slice1 = torch.nn.Sequential()
            self.slice2 = torch.nn.Sequential()
            self.slice3 = torch.nn.Sequential()
            self.slice4 = torch.nn.Sequential()
            self.slice5 = torch.nn.Sequential()
            for x in range(2):
                self.slice1.add_module(str(x), vgg_pretrained_features[x])
            for x in range(2, 7):
                self.slice2.add_module(str(x), vgg_pretrained_features[x])
            for x in range(7, 12):
                self.slice3.add_module(str(x), vgg_pretrained_features[x])
            for x in range(12, 21):
                self.slice4.add_module(str(x), vgg_pretrained_features[x])
            for x in range(21, 36):
                self.slice5.add_module(str(x), vgg_pretrained_features[x])

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, x):
        if self.layer == 'four':
            h = self.slice(x)

        elif self.layer == 'five':
            h = self.slice(x)

        else:
            h_relu1 = self.slice1(x)
            h_relu2 = self.slice2(h_relu1)
            h_relu3 = self.slice3(h_relu2)
            h_relu4 = self.slice4(h_relu3)
            h_relu5 = self.slice5(h_relu4)

        return [h_relu2, h_relu3, h_relu4, h_relu5]


class SoftAdaIN(nn.Module):
    def __init__(self, ch):
        super(SoftAdaIN, self).__init__()

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(ch*2, ch*2),
            nn.ReLU(),
            nn.Linear(ch*2, ch),
            nn.ReLU()
        )

        self.fade = nn.Sequential(
            nn.Conv2d(ch, ch, 1, 1, 0),
            nn.ReLU(),
            nn.Conv2d(ch, ch, 3, 1, 1)
        )

    def forward(self, xp, xc):
        xp_gap = self.gap(xp).squeeze(3).squeeze(2)
        xc_gap = self.gap(xc).squeeze(3).squeeze(2)

        h = torch.cat([xp_gap, xc_gap], dim=1)
        h = self.fc(h).unsqueeze(2).unsqueeze(3)

        h = adain(xp, xc, h)
        hout = self.fade(h)

        return hout, h


class RenderingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up=False):
        super(RenderingBlock, self).__init__()

        self.up = up
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear")
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.ReLU()
        )

    def forward(self, x, s=None):
        if self.up:
            x = self.upsample(x)

        if s is None:
            h = self.block(x)
        else:
            h = self.block(torch.cat([x, s], dim=1))

        return h


class RenderNetwork(nn.Module):
    def __init__(self, base=64):
        super(RenderNetwork, self).__init__()

        self.dec0 = RenderingBlock(base*8, base*8)
        self.dec1 = RenderingBlock(base*16, base*4, up=True)
        self.dec2 = RenderingBlock(base*8, base*2, up=True)
        self.dec3 = RenderingBlock(base*4, base, up=True)
        self.dec4 = RenderingBlock(base, base, up=True)

        self.dout = nn.Sequential(
            nn.Conv2d(base, 3, 1, 1, 0),
            nn.Tanh()
        )

    def forward(self, sa_list):
        h = self.dec0(sa_list[0])
        h = self.dec1(h, sa_list[1])
        h = self.dec2(h, sa_list[2])
        h = self.dec3(h, sa_list[3])
        h = self.dec4(h)
        h = self.dout(h)

        return h


class CartoonRenderer(nn.Module):
    def __init__(self, base=64):
        super(CartoonRenderer, self).__init__()
        self.fe = Vgg19(requires_grad=False)

        self.sa0 = SoftAdaIN(ch=base*8)
        self.sa1 = SoftAdaIN(ch=base*8)
        self.sa2 = SoftAdaIN(ch=base*4)
        self.sa3 = SoftAdaIN(ch=base*2)

        self.rn = RenderNetwork()

        init_weights(self.sa0)
        init_weights(self.sa1)
        init_weights(self.sa2)
        init_weights(self.sa3)
        init_weights(self.rn)

    def forward(self, xp, xc):
        xp_feat = self.fe(xp)
        xc_feat = self.fe(xc)

        sa0, out0 = self.sa0(xp_feat[3], xc_feat[3])
        sa1, out1 = self.sa1(xp_feat[2], xc_feat[2])
        sa2, out2 = self.sa2(xp_feat[1], xc_feat[1])
        sa3, out3 = self.sa3(xp_feat[0], xc_feat[0])
        h_list = [sa0, sa1, sa2, sa3]
        sa_list = [out0, out1, out2, out3]

        h = self.rn(h_list)
        y_feat = self.fe(h)

        return h, xc_feat, sa_list[::-1], y_feat


class CIL(nn.Module):
    def __init__(self, in_ch, out_ch, kernel=3, stride=1, pad=1, norm=False):
        super(CIL, self).__init__()

        if norm:
            self.cia = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
                nn.InstanceNorm2d(out_ch),
                nn.LeakyReLU()
            )

        else:
            self.cia = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel, stride, pad),
                nn.LeakyReLU()
            )

    def forward(self, x):
        return self.cia(x)


class Discriminator(nn.Module):
    def __init__(self, base=64):
        super(Discriminator, self).__init__()
        self.cnns = nn.ModuleList()
        for _ in range(3):
            self.cnns.append(self._make_nets(base))
        self.down = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def _make_nets(self, base):
        model = nn.Sequential(
            CIL(3, base, 4, 2, 1),
            CIL(base, base*2, 4, 2, 1),
            CIL(base*2, base*4, 4, 2, 1),
            CIL(base*4, base*8, 4, 2, 1),
            nn.Conv2d(base*8, 1, 1, 1, 0)
        )

        init_weights(model)

        return model

    def forward(self, x):
        outputs = []
        for model in self.cnns:
            h = model(x)
            outputs.append(h)
            x = self.down(x)

        return outputs
