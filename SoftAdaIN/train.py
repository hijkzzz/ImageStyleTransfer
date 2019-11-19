import torch
import torch.nn as nn
import numpy as np
import argparse

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from model import CartoonRenderer, Discriminator, calc_mean_std
from dataset import CRDataset, CollateFn

softplus = nn.Softplus()


def feature_normalize(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)

    normalized_feat = (feat - mean.expand(size)) / std.expand(size)

    return normalized_feat


def adversarial_loss_dis(discriminator, y, t):
    fake_list = discriminator(y.detach())
    real_list = discriminator(t)

    sum_loss = 0
    for fake, real in zip(fake_list, real_list):
        loss = torch.mean(softplus(fake)) + torch.mean(softplus(-real))
        sum_loss += loss

    return sum_loss


def adversarial_loss_gen(discriminator, y):
    fake_list = discriminator(y)

    sum_loss = 0
    for fake in fake_list:
        loss = torch.mean(softplus(-fake))
        sum_loss += loss

    return sum_loss


def content_loss(y_list, t_list):
    sum_loss = 0
    for y, t in zip(y_list, t_list):
        norm_y, norm_t = feature_normalize(y), feature_normalize(t)
        loss = torch.mean(torch.abs(norm_y - norm_t))
        sum_loss += loss

    return sum_loss


def style_loss(y_list, t_list):
    sum_loss = 0
    for y, t in zip(y_list, t_list):
        mean_y, std_y = calc_mean_std(y)
        mean_t, std_t = calc_mean_std(t)

        mean_loss = torch.mean((mean_y - mean_t) ** 2)
        std_loss = torch.mean((std_y - std_t) ** 2)

        sum_loss += mean_loss
        sum_loss += std_loss

    return 20 * sum_loss


def reconstruction_loss(y, t):
    return 0.0001 * torch.mean((y-t) ** 2)


def train(epochs, batchsize, interval, c_path, s_path, modeldir):
    # Dataset definition
    dataset = CRDataset(c_path, s_path)
    collator = CollateFn()

    # Model definition
    generator = CartoonRenderer()
    generator.cuda()
    generator.train()
    gen_opt = torch.optim.Adam(generator.parameters(), lr=0.0001)

    discriminator = Discriminator()
    discriminator.cuda()
    discriminator.train()
    dis_opt = torch.optim.Adam(discriminator.parameters(), lr=0.0001)

    iterations = 0

    for epoch in range(epochs):
        dataloader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=collator)
        dataloader = tqdm(dataloader)

        for i, data in enumerate(dataloader):
            iterations += 1
            c, s = data

            y, _, _, _ = generator(c, s)
            dis_loss = adversarial_loss_dis(discriminator, y, s)

            dis_opt.zero_grad()
            dis_loss.backward()
            dis_opt.step()

            y, c_feat, sa_list, y_feat = generator(c, s)
            y_c, _, _, _ = generator(c, c)
            y_s, _, _, _ = generator(s, s)

            gen_loss = adversarial_loss_gen(discriminator, y)
            gen_loss += reconstruction_loss(y_c, c)
            gen_loss += reconstruction_loss(y_s, s)
            gen_loss += content_loss(sa_list, y_feat)
            gen_loss += style_loss(c_feat, y_feat)

            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            if iterations % interval == 1:
                torch.save(generator.state_dict(), f"{modeldir}/model_{iterations}.pt")

            print(f"iter: {iterations} dis loss: {dis_loss.data} gen loss: {gen_loss.data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Soft AdaIN")
    parser.add_argument("--e", type=int, default=1000, help="the number of epochs")
    parser.add_argument("--b", type=int, default=16, help="batch size")
    parser.add_argument("--i", type=int, default=2000, help="the interval of snapshot")
    args = parser.parse_args()

    modeldir = Path("./modeldir")
    modeldir.mkdir(exist_ok=True)

    c_path = Path('./coco2017/train2017/')
    s_path = Path('./cell/')

    train(args.e, args.b, args.i, c_path, s_path, modeldir)
