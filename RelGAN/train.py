import numpy as np
import chainer
import argparse
import chainer.functions as F
import chainer.links as L

from pathlib import Path
from dataset import DatasetLoader
from model import Generator, Discriminator
from utils import set_optimizer, call_zeros, call_ones
from chainer import serializers, cuda

xp = cuda.cupy
cuda.get_device(0).use()


class RelGANLossFunction:
    def __init__(self):
        pass

    @staticmethod
    def zero_centered_gradient_penalty_fake(fake, y):
        grad, = chainer.grad([fake], [y], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        zeros = call_zeros(grad)

        loss = 10 * F.mean_squared_error(grad, zeros)

        return loss

    @staticmethod
    def zero_centered_gradient_penalty_real(discriminator, t):
        t = chainer.Variable(t.data)
        real = discriminator(t, y=None, label=None, method="adv")

        grad, = chainer.grad([real], [t], enable_double_backprop=True)
        grad = F.sqrt(F.batch_l2_norm_squared(grad))
        zeros = call_zeros(grad)

        loss = 10 * F.mean_squared_error(grad, zeros)

        return loss

    def adversarial_loss_dis(self, discriminator, y, t):
        fake = discriminator(y, y=None, label=None, method="adv")
        real = discriminator(t, y=None, label=None, method="adv")

        loss = self.zero_centered_gradient_penalty_fake(fake, y)
        loss += self.zero_centered_gradient_penalty_real(discriminator, t)

        return F.mean(F.relu(1. - real)) + F.mean(F.relu(1. + fake)) + loss

    @staticmethod
    def adversarial_loss_gen(discriminator, y):
        fake = discriminator(y, y=None, label=None, method="adv")

        return -F.mean(fake)

    @staticmethod
    def interpolation_loss_dis(discriminator, y0, y1, yalpha, alpha, flag):
        fake_y0 = discriminator(y0, y=None, label=None, method="inp")
        fake_y1 = discriminator(y1, y=None, label=None, method="inp")
        fake_yalpha = discriminator(yalpha, y=None, label=None, method="inp")

        zeros = call_zeros(fake_y0)

        if flag == 0:
            loss = F.mean_squared_error(fake_y0, zeros)
            loss += F.mean_squared_error(fake_yalpha, alpha)

        else:
            loss = F.mean_squared_error(fake_y1, zeros)
            loss += F.mean_squared_error(fake_yalpha, (1 - alpha))

        return loss

    @staticmethod
    def interpolation_loss_gen(discriminator, yalpha):
        fake_yalpha = discriminator(yalpha, y=None, label=None, method="inp")

        zeros = call_zeros(fake_yalpha)

        return F.mean_squared_error(fake_yalpha, zeros)

    @staticmethod
    def matching_loss_dis(discriminator, x, fake, y, z, v1, v2, v3):
        sr = discriminator(x, y, v1, method="mat")
        sf = discriminator(x, fake, v1, method="mat")
        sw0 = discriminator(z, y, v1, method="mat")
        sw1 = discriminator(x, y, v2, method="mat")
        sw2 = discriminator(x, y, v3, method="mat")
        sw3 = discriminator(x, z, v1, method="mat")

        zeros = call_zeros(sr)
        ones = call_ones(sr)

        loss = F.mean_squared_error(sr, ones)
        loss += F.mean_squared_error(sf, zeros)
        loss += F.mean_squared_error(sw0, zeros)
        loss += F.mean_squared_error(sw1, zeros)
        loss += F.mean_squared_error(sw2, zeros)
        loss += F.mean_squared_error(sw3, zeros)

        return loss

    @staticmethod
    def matching_loss_gen(discriminator, x, t, v1):
        sf = discriminator(x, t, v1, method="mat")

        ones = call_ones(sf)

        loss = F.mean_squared_error(sf, ones)

        return loss


def train(epochs,
          iterations,
          batchsize,
          data_path,
          modeldir,
          extension,
          img_size,
          learning_rate,
          beta1,
          weight_decay
          ):

    # Dataset definition
    dataset = DatasetLoader(data_path, extension, img_size)

    # Model & Optimizer definition
    generator = Generator(dataset.number)
    generator.to_gpu()
    gen_opt = set_optimizer(generator, learning_rate, beta1, weight_decay)

    discriminator = Discriminator(dataset.number)
    discriminator.to_gpu()
    dis_opt = set_optimizer(discriminator, learning_rate, beta1, weight_decay)

    # Loss Function definition
    lossfunc = RelGANLossFunction()

    for epoch in range(epochs):
        sum_dis_loss = 0
        sum_gen_loss = 0
        for batch in range(0, iterations, batchsize):
            x, x_label, y, y_label, z, z_label = dataset.train(batchsize)

            # Discriminator update
            # Adversairal loss
            a = y_label - x_label
            fake = generator(x, a)
            fake.unchain_backward()
            loss = lossfunc.adversarial_loss_dis(discriminator, fake, y)

            # Interpolation loss
            rnd = np.random.randint(2)
            if rnd == 0:
                alpha = xp.random.uniform(0, 0.5, size=batchsize)
            else:
                alpha = xp.random.uniform(0.5, 1.0, size=batchsize)
            alpha = chainer.as_variable(alpha.astype(xp.float32))
            alpha = F.tile(F.expand_dims(alpha, axis=1), (1, dataset.number))

            fake_0 = generator(x, y_label - y_label)
            fake_1 = generator(x, alpha*a)
            fake_0.unchain_backward()
            fake_1.unchain_backward()
            loss += 10 * lossfunc.interpolation_loss_dis(discriminator, fake_0, fake, fake_1, alpha, rnd)

            # Matching loss
            v2 = y_label - z_label
            v3 = z_label - x_label

            loss += lossfunc.matching_loss_dis(discriminator, x, fake, y, z, a, v2, v3)

            discriminator.cleargrads()
            loss.backward()
            dis_opt.update()
            loss.unchain_backward()

            sum_dis_loss += loss.data

            # Generator update
            # Adversarial loss
            fake = generator(x, a)
            loss = lossfunc.adversarial_loss_gen(discriminator, fake)

            # Interpolation loss
            rnd = np.random.randint(2)
            if rnd == 0:
                alpha = xp.random.uniform(0, 0.5, size=batchsize)
            else:
                alpha = xp.random.uniform(0.5, 1.0, size=batchsize)
            alpha = chainer.as_variable(alpha.astype(xp.float32))
            alpha = F.tile(F.expand_dims(alpha, axis=1), (1, dataset.number))

            fake_alpha = generator(x, alpha*a)
            loss += 10 * lossfunc.interpolation_loss_gen(discriminator, fake_alpha)

            # Matching loss
            loss += lossfunc.matching_loss_gen(discriminator, x, fake, a)

            # Cycle-consistency loss
            cyc = generator(fake, -a)
            loss += 10 * F.mean_absolute_error(cyc, x)

            # Self-reconstruction loss
            fake_0 = generator(x, y_label - y_label)
            loss += 10 * F.mean_absolute_error(fake_0, x)

            generator.cleargrads()
            loss.backward()
            gen_opt.update()
            loss.unchain_backward()

            sum_gen_loss += loss.data

            if batch == 0:
                serializers.save_npz(f"{modeldir}/generator_{epoch}.model", generator)

        print(f"epoch: {epoch} disloss: {sum_dis_loss/iterations} genloss: {sum_gen_loss/iterations}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RelGAN")
    parser.add_argument("--e", type=int, default=1000, help="the number of epochs")
    parser.add_argument("--i", type=int, default=2000, help="the number of iterations")
    parser.add_argument("--b", type=int, default=16, help="batch size")
    parser.add_argument("--modeldir", type=Path, default="modeldir", help="model output directory")
    parser.add_argument("--ext", type=str, default=".png", help="extension of training images")
    parser.add_argument("--size", type=int, default=128, help="the size of training images")
    parser.add_argument("--lr", type=float, default=0.00005, help="learning rate of Adam")
    parser.add_argument("--b1", type=float, default=0.9, help="beta1 of Adam")
    parser.add_argument("--wd", type=float, default=0.00001, help="weight decay of optimizer")
    parser.add_argument("--path", type=Path, help="path containing image paths")

    args = parser.parse_args()

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.i, args.b, args.path, modeldir, args.ext, args.size, args.lr, args.b1, args.wd)
