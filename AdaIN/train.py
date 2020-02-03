import chainer
import argparse
import chainer.functions as F
import numpy as np

from pathlib import Path
from chainer import serializers, cuda
from model import Decoder, VGG, adain, calc_mean_std
from dataset import DatasetLoader
from utils import set_optimizer
from visualize import Visualizer

xp = cuda.cupy
cuda.get_device(0).use()


class AdaINLossFunctionCalculator:
    def __init__(self):
        pass

    @staticmethod
    def content_loss(y, t):
        return F.mean_absolute_error(y, t)

    @staticmethod
    def style_loss(y_feat_list, t_feat_list):
        sum_loss = 0
        for y_feat, t_feat in zip(y_feat_list, t_feat_list):
            y_std, y_mean = calc_mean_std(y_feat)
            t_std, t_mean = calc_mean_std(t_feat)

        sum_loss += 10 * F.mean_squared_error(y_std, t_std) + F.mean_squared_error(y_mean, t_mean)

        return sum_loss


def train(epochs,
          iterations,
          batchsize,
          validsize,
          outdir,
          modeldir,
          con_path,
          sty_path,
          extension,
          coord_size,
          crop_size,
          alpha,
          learning_rate,
          beta1):
    
    # Dataset definition
    dataloader = DatasetLoader(con_path, sty_path, extension, coord_size, crop_size)
    print(dataloader)
    con_valid, sty_valid = dataloader.valid(validsize)

    # Mode & Optimizer defnition
    decoder = Decoder()
    decoder.to_gpu()
    dec_opt = set_optimizer(decoder, learning_rate, beta1)

    vgg = VGG()
    vgg.to_gpu()
    vgg_opt = set_optimizer(vgg, learning_rate, beta1)
    vgg.base.disable_update()

    # Loss Function definition
    lossfunc = AdaINLossFunctionCalculator()

    # Visualizer definition
    visualizer = Visualizer()

    for epoch in range(epochs):
        sum_loss = 0
        for batch in range(0, iterations, batchsize):
            con, sty = dataloader.train(batchsize)

            style_feat_list = vgg(sty)
            content_feat = vgg(con)[-1]

            t = adain(content_feat, style_feat_list[-1])
            t = alpha * t + (1.0 - alpha) * content_feat

            g_t = decoder(t)
            g_t_feat_list = vgg(g_t)

            loss = lossfunc.content_loss(g_t_feat_list[-1], t)
            loss += lossfunc.style_loss(style_feat_list, g_t_feat_list)

            decoder.cleargrads()
            vgg.cleargrads()
            loss.backward()
            dec_opt.update()
            vgg_opt.update()

            sum_loss += loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/decoder_{epoch}.model", decoder)

                with chainer.using_config("train", False):
                    style_feat_list = vgg(sty_valid)
                    content_feat = vgg(con_valid)[-1]

                    t = adain(content_feat, style_feat_list[-1])
                    t = alpha * t + (1 - alpha) * content_feat
                    g_t = decoder(t)

                g_t = g_t.data.get()
                con = con_valid.data.get()
                sty = sty_valid.data.get()

                visualizer(con, sty, g_t, outdir, epoch, validsize)

        print(f"epoch: {epoch}")
        print(f"loss: {sum_loss / iterations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaIN")
    parser.add_argument("--e", type=int, default=2000, help="the number of epochs")
    parser.add_argument("--i", type=int, default=20000, help="the interval of snapshot")
    parser.add_argument("--b", type=int, default=16, help="batch size")
    parser.add_argument("--v", type=int, default=12, help="valid size")
    parser.add_argument("--outdir", type=Path, default="outdir", help="output directory")
    parser.add_argument("--modeldir", type=Path, default="modeldir", help="model output directory")
    parser.add_argument("--ext", type=str, default=".jpg", help="extension of training image")
    parser.add_argument("--coord", type=int, default=512, help="the coordinate size of image")
    parser.add_argument("--crop", type=int, default=256, help="the cropped size of image")
    parser.add_argument("--lr", type=float, default=0.0001, help="leraning rate of optimizer")
    parser.add_argument("--b1", type=float, default=0.5, help="beta1 of Adam")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha of AdaIN")
    parser.add_argument("--con_path", type=Path, help="path containing contain images")
    parser.add_argument("--sty_path", type=Path, help="path containing style images")

    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, args.v, outdir, modeldir, args.con_path, args.sty_path, args.ext,
          args.coord, args.crop, args.alpha, args.lr, args.b1)
