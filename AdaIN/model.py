import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np

from chainer import Chain,cuda

xp = cuda.cupy


class CR(Chain):
    def __init__(self, in_ch, out_ch, sample ="up", activation=F.relu):
        super(CR, self).__init__()
        w = chainer.initializers.Normal(0.02)
        self.activation = activation
        self.sample = sample

        with self.init_scope():
            self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)

    def __call__(self, x):
        h = x
        if self.activation is not None:
            h = self.activation(self.c0(h))
        if self.sample == "up":
            h = F.unpooling_2d(h, 2, 2, 0, cover_all=False)

        return h


class Decoder(Chain):
    def __init__(self):
        super(Decoder, self).__init__()
        w = chainer.initializers.Normal(0.02)
        with self.init_scope():
            self.cr0 = CR(512, 256, sample="up")
            self.cr1 = CR(256, 256, sample="same")
            self.cr2 = CR(256, 256, sample="same")
            self.cr3 = CR(256, 256, sample="same")
            self.cr4 = CR(256, 128, sample="up")
            self.cr5 = CR(128, 128, sample="same")
            self.cr6 = CR(128, 64, sample="up")
            self.cr7 = CR(64, 64, sample="same")
            self.c0 = L.Convolution2D(64, 3, 3, 1, 1, initialW=w)

    def __call__(self, x):
        h = self.cr0(x)
        h = self.cr1(h)
        h = self.cr2(h)
        h = self.cr3(h)
        h = self.cr4(h)
        h = self.cr5(h)
        h = self.cr6(h)
        h = self.cr7(h)
        h = F.tanh(self.c0(h))

        return h


class VGG(Chain):
    def __init__(self, last_only=False):
        super(VGG, self).__init__()
        self.last_only = last_only
        with self.init_scope():
            self.base = L.VGG19Layers()

    def __call__(self,x, last_only=False):
        h1 = self.base(x, layers=["conv1_1"])["conv1_1"]
        h2 = self.base(x, layers=["conv2_1"])["conv2_1"]
        h3 = self.base(x, layers=["conv3_1"])["conv3_1"]
        h4 = self.base(x, layers=["conv4_1"])["conv4_1"]

        return [h1, h2, h3, h4]


def calc_mean_std(feature, eps=1e-5):
    batch, channels, _, _ = feature.shape
    feature_map = feature.reshape(batch, channels, -1)
    widhei = feature_map.shape[2]
    feature_mean = F.mean(feature_map, axis=2)
    feature_var = (feature_map - F.tile(F.expand_dims(feature_mean, axis=2), (1, 1, widhei))) ** 2
    feature_var = F.mean(feature_var, axis=2)
    feature_std = F.sqrt(feature_var + eps).reshape(batch, channels, 1, 1)
    feature_mean = feature_mean.reshape(batch, channels, 1, 1)

    return feature_std, feature_mean


def adain(content_feature, style_feature):
    shape = content_feature.shape
    style_std, style_mean = calc_mean_std(style_feature)
    style_mean = F.broadcast_to(style_mean, shape=shape)
    style_std = F.broadcast_to(style_std, shape=shape)

    content_std, content_mean = calc_mean_std(content_feature)
    content_mean = F.broadcast_to(content_mean, shape=shape)
    content_std = F.broadcast_to(content_std, shape=shape)
    normalized_feat = (content_feature - content_mean) / content_std

    return normalized_feat * style_std + style_mean
