import chainer
import chainer.links as L
import chainer.functions as F
import numpy as np
import argparse

from pathlib import Path
from chainer import cuda, optimizers, serializers
from model import Generator, Discriminator
from dataset import DatasetLoader
from utils import set_optimizer
from visualize import Visualizer

xp=cuda.cupy
cuda.get_device(0).use()


class InstaGANLossFunction:
    def __init__(self):
        pass

    @staticmethod
    def adversarial_dis_loss(discriminator, y, y_mask, t, t_mask):
        y_dis = discriminator(y, y_mask)
        t_dis = discriminator(t, t_mask)

        return F.mean(F.softplus(y_dis)) + F.mean(F.softplus(-t_dis))

    @staticmethod
    def adversarial_gen_loss(discriminator, y, y_mask):
        y_dis = discriminator(y, y_mask)

        return F.mean(F.softplus(-y_dis))

    @staticmethod
    def cycle_consistency_loss(y, y_mask, t, t_mask):
        loss = F.mean_absolute_error(y, t)
        loss += F.mean_absolute_error(y_mask, t_mask)

        return 10.0 * loss

    @staticmethod
    def identity_mapping_loss(y, y_mask, t, t_mask):
        loss = F.mean_absolute_error(y, t)
        loss += F.mean_absolute_error(y_mask, t_mask)

        return 10.0 * loss

    @staticmethod
    def context_preserving_loss(y, y_mask, t, t_mask):
        weight = F.tile(xp.ones_like(t_mask) - t_mask * y_mask, (1, 3, 1, 1))
        loss = F.mean_absolute_error(weight * t, weight * y)

        return 10.0 * loss


def train(epochs,
          iterations,
          batchsize,
          validsize,
          src_path,
          tgt_path,
          extension,
          img_size,
          outdir,
          modeldir,
          lr_dis,
          lr_gen,
          beta1,
          beta2):

    # Dataset definition
    dataset = DatasetLoader(src_path, tgt_path, extension, img_size)
    print(dataset)
    x_val, x_mask_val, y_val, y_mask_val = dataset.valid(validsize)

    # Model & Optimizer definition
    generator_xy = Generator()
    generator_xy.to_gpu()
    gen_xy_opt = set_optimizer(generator_xy, lr_gen, beta1, beta2)

    generator_yx = Generator()
    generator_yx.to_gpu()
    gen_yx_opt = set_optimizer(generator_yx, lr_gen, beta1, beta2)

    discriminator_y = Discriminator()
    discriminator_y.to_gpu()
    dis_y_opt = set_optimizer(discriminator_y, lr_dis, beta1, beta2)

    discriminator_x = Discriminator()
    discriminator_x.to_gpu()
    dis_x_opt = set_optimizer(discriminator_x, lr_dis, beta1, beta2)

    # Loss Function definition
    lossfunc = InstaGANLossFunction()

    # Visualizer definition
    visualize = Visualizer()

    for epoch in range(epochs):
        sum_gen_loss = 0
        sum_dis_loss = 0

        for batch in range(0, iterations, batchsize):
            x, x_mask, y, y_mask = dataset.train(batchsize)

            # discriminator update
            xy, xy_mask = generator_xy(x, x_mask)
            yx, yx_mask = generator_yx(y, y_mask)

            xy.unchain_backward()
            xy_mask.unchain_backward()
            yx.unchain_backward()
            yx_mask.unchain_backward()

            dis_loss = lossfunc.adversarial_dis_loss(discriminator_y, xy, xy_mask, y, y_mask)
            dis_loss += lossfunc.adversarial_dis_loss(discriminator_x, yx, yx_mask, x, x_mask)

            discriminator_y.cleargrads()
            discriminator_x.cleargrads()
            dis_loss.backward()
            dis_y_opt.update()
            dis_x_opt.update()

            sum_dis_loss += dis_loss.data

            # generator update
            xy, xy_mask = generator_xy(x, x_mask)
            yx, yx_mask = generator_yx(y, y_mask)

            xyx, xyx_mask = generator_yx(xy, xy_mask)
            yxy, yxy_mask = generator_xy(yx, yx_mask)

            x_id, x_mask_id = generator_yx(x, x_mask)
            y_id, y_mask_id = generator_xy(y, y_mask)

            gen_loss = lossfunc.adversarial_gen_loss(discriminator_y, xy, xy_mask)
            gen_loss += lossfunc.adversarial_gen_loss(discriminator_x, yx, yx_mask)

            gen_loss += lossfunc.cycle_consistency_loss(xyx, xyx_mask, x, x_mask)
            gen_loss += lossfunc.cycle_consistency_loss(yxy, yxy_mask, y, y_mask)

            gen_loss += lossfunc.identity_mapping_loss(x_id, x_mask_id, x, x_mask)
            gen_loss += lossfunc.identity_mapping_loss(y_id, y_mask_id, y, y_mask)

            gen_loss += lossfunc.context_preserving_loss(xy, xy_mask, x, x_mask)
            gen_loss += lossfunc.context_preserving_loss(yx, yx_mask, y, y_mask)

            generator_xy.cleargrads()
            generator_yx.cleargrads()
            gen_loss.backward()
            gen_xy_opt.update()
            gen_yx_opt.update()

            sum_gen_loss += gen_loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_xy_{epoch}.model", generator_xy)
                serializers.save_npz(f"{modeldir}/generator_yx_{epoch}.model", generator_yx)

                xy, xy_mask = generator_xy(x_val, x_mask_val)
                yx, yx_mask = generator_yx(y_val, y_mask_val)

                x = x_val.data.get()
                x_mask = x_mask_val.data.get()
                xy = xy.data.get()
                xy_mask = xy_mask.data.get()

                visualize(x, x_mask, xy, xy_mask, outdir, epoch, validsize, switch="mtot")

                y = y_val.data.get()
                y_mask = y_mask_val.data.get()
                yx = yx.data.get()
                yx_mask = yx_mask.data.get()

                visualize(y, y_mask, yx, yx_mask, outdir, epoch, validsize, switch="ttom")

        print(f"epoch: {epoch}")
        print(f"dis loss: {sum_dis_loss / iterations} gen loss: {sum_gen_loss / iterations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InstaGAN")
    parser.add_argument("--e", type=int, default=1000, help="the number of epochs")
    parser.add_argument("--i", type=int, default=2000, help="the interval of snapshot")
    parser.add_argument("--b", type=int, default=16, help="batch size")
    parser.add_argument("--v", type=int, default=4, help="valid size")
    parser.add_argument("--ext", type=str, default=".png", help="extension of training images")
    parser.add_argument("--size", type=int, default=128, help="the size of training images")
    parser.add_argument("--outdir", type=Path, default='outdir', help="output directory")
    parser.add_argument("--modeldir", type=Path, default='modeldir', help="model output directory")
    parser.add_argument("--lrdis", type=float, default=0.0001, help="discriminator alpha of Adam")
    parser.add_argument("--lrgen", type=float, default=0.0002, help="generator alpha of Adam")
    parser.add_argument("--b1", type=float, default=0.5, help="beta1 of Adam")
    parser.add_argument("--b2", type=float, default=0.999, help="beta2 of Adam")
    parser.add_argument("--src_path", type=Path, help="path which contains source images")
    parser.add_argument("--tgt_path", type=Path, help="path which contains target images")

    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, args.v, args.src_path, args.tgt_path, args.ext, args.size,
          args.outdir, args.modeldir, args.lrdis, args.lrgen, args.b1, args.b2)
