import chainer
import chainer.functions as F
import argparse
import numpy as np

from pathlib import Path
from chainer import serializers, cuda
from model import Generator, Discriminator
from utils import set_optimizer
from dataset import DatasetLoader
from visualize import Visualization

xp = cuda.cupy
cuda.get_device(0).use()


class CycleGANLossCalculator:
    def __init__(self):
        pass

    @staticmethod
    def dis_loss(discriminator, y, t):
        y_dis = discriminator(y)
        t_dis = discriminator(t)

        loss = F.mean(F.softplus(-t_dis)) + F.mean(F.softplus(y_dis))

        return loss

    @staticmethod
    def gen_loss(discriminator, y):
        y_dis = discriminator(y)

        loss = F.mean(F.softplus(-y_dis))

        return loss

    @staticmethod
    def cycle_consitency_loss(y, t):
        return 10 * F.mean_absolute_error(y, t)

    @staticmethod
    def identity_mapping_loss(y, t):
        return 10 * F.mean_absolute_error(y, t)


def train(epochs,
          iterations,
          batchsize,
          validsize,
          outdir,
          modeldir,
          src_path,
          tgt_path,
          extension,
          img_size,
          learning_rate,
          beta1
          ):

    # Dataset definition
    dataloader = DatasetLoader(src_path, tgt_path, extension, img_size)
    print(dataloader)
    src_val = dataloader.valid(validsize)

    # Model & Optimizer definition
    generator_xy = Generator()
    generator_xy.to_gpu()
    gen_xy_opt = set_optimizer(generator_xy, learning_rate, beta1)

    generator_yx = Generator()
    generator_yx.to_gpu()
    gen_yx_opt = set_optimizer(generator_yx, learning_rate, beta1)

    discriminator_y = Discriminator()
    discriminator_y.to_gpu()
    dis_y_opt = set_optimizer(discriminator_y, learning_rate, beta1)

    discriminator_x = Discriminator()
    discriminator_x.to_gpu()
    dis_x_opt = set_optimizer(discriminator_x, learning_rate, beta1)

    # LossFunction definition
    lossfunc = CycleGANLossCalculator()

    # Visualization
    visualizer = Visualization()

    for epoch in range(epochs):
        sum_gen_loss = 0
        sum_dis_loss = 0
        for batch in range(0, iterations, batchsize):
            x, y = dataloader.train(batchsize)

            # Discriminator update
            xy = generator_xy(x)
            yx = generator_yx(y)

            xy.unchain_backward()
            yx.unchain_backward()

            dis_loss_xy = lossfunc.dis_loss(discriminator_y, xy, y)
            dis_loss_yx = lossfunc.dis_loss(discriminator_x, yx, x)

            dis_loss = dis_loss_xy + dis_loss_yx

            discriminator_x.cleargrads()
            discriminator_y.cleargrads()
            dis_loss.backward()
            dis_x_opt.update()
            dis_y_opt.update()

            sum_dis_loss += dis_loss.data

            # Generator update
            xy = generator_xy(x)
            yx = generator_yx(y)

            xyx = generator_yx(xy)
            yxy = generator_xy(yx)

            y_id = generator_xy(y)
            x_id = generator_yx(x)

            # adversarial loss
            gen_loss_xy = lossfunc.gen_loss(discriminator_y, xy)
            gen_loss_yx = lossfunc.gen_loss(discriminator_x, yx)

            # cycle-consitency loss
            cycle_y = lossfunc.cycle_consitency_loss(yxy, y)
            cycle_x = lossfunc.cycle_consitency_loss(xyx, x)

            # identity mapping loss
            identity_y = lossfunc.identity_mapping_loss(y_id, y)
            identity_x = lossfunc.identity_mapping_loss(x_id, x)

            gen_loss = gen_loss_xy + gen_loss_yx + cycle_x + cycle_y + identity_x + identity_y

            generator_xy.cleargrads()
            generator_yx.cleargrads()
            gen_loss.backward()
            gen_xy_opt.update()
            gen_yx_opt.update()

            sum_gen_loss += gen_loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_xy_{epoch}.model", generator_xy)
                serializers.save_npz(f"{modeldir}/generator_yx_{epoch}.model", generator_yx)

                with chainer.using_config('train', False):
                    tgt = generator_xy(src_val)

                src = src_val.data.get()
                tgt = tgt.data.get()

                visualizer(src, tgt, outdir, epoch, validsize)

        print(f"epoch: {epoch}")
        print(F"dis loss: {sum_dis_loss/iterations} gen loss: {sum_gen_loss/iterations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN")
    parser.add_argument("--e", type=int, default=1000, help="the number of epochs")
    parser.add_argument("--i", type=int, default=20000, help="the interval of snapshot")
    parser.add_argument("--b", type=int, default=16, help="batch size")
    parser.add_argument("--v", type=int, default=8, help="valid size")
    parser.add_argument("--outdir", type=Path, default="validdir", help="output directory")
    parser.add_argument("--modeldir", type=Path, default="modeldir", help="model output directory")
    parser.add_argument('--ext', type=str, default=".png", help="the extension of training data")
    parser.add_argument("--size", type=int, default=128, help="the size of training data")
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate of Adam")
    parser.add_argument("--b1", type=float, default=0.5, help="beta1 of Adam")
    parser.add_argument("--src_path", type=Path, help="path which contains source data")
    parser.add_argument("--tgt_path", type=Path, help="path which contains target data")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, args.v, outdir, modeldir, args.src_path, args.tgt_path,
          args.ext, args.size, args.lr, args.b1)
