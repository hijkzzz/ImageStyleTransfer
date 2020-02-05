import torch
import torch.nn as nn
import argparse

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import DataLoader
from dataset import UGATITDataset, ImageCollate
from model import Generator, Discriminator, RhoClipper, GlobalDiscriminator

mseloss = nn.MSELoss()
l1loss = nn.L1Loss()
bceloss = nn.BCEWithLogitsLoss()


def discriminator_loss(fake, real, fake_logit, real_logit):
    adv_loss = mseloss(real, torch.ones_like(real)) + mseloss(fake, torch.zeros_like(fake))
    adv_loss += mseloss(real_logit, torch.ones_like(real_logit)) + mseloss(fake_logit, torch.zeros_like(fake_logit))

    return adv_loss


def generator_loss(fake, real, fake_logit, fake_g_logit,
                   fake_xyx, fake_id, fake_id_logit):
    adv_loss = mseloss(fake, torch.ones_like(fake))
    adv_loss += mseloss(fake_logit, torch.ones_like(fake_logit))

    cycle_loss = l1loss(fake_xyx, real)
    identity_loss = l1loss(fake_id, real)

    cam_loss = bceloss(fake_g_logit, torch.ones_like(fake_g_logit))
    cam_loss += bceloss(fake_id_logit, torch.zeros_like(fake_id_logit))

    return adv_loss + 10 * cycle_loss + 10 * identity_loss + 1000 * cam_loss


def train(epochs,
          batchsize,
          interval,
          modeldir,
          img_size,
          src_path,
          tgt_path,
          learning_rate,
          beta1,
          beta2):

    # Dataset definition
    dataset = UGATITDataset(src_path, tgt_path)
    print(dataset)
    collator = ImageCollate(img_size)

    # Model & Optimizer definition
    generator_st = Generator()
    generator_st.cuda()
    generator_st.train()
    optim_gen_st = torch.optim.Adam(generator_st.parameters(), lr=learning_rate, betas=(beta1, beta2))

    generator_ts = Generator()
    generator_ts.cuda()
    generator_ts.train()
    optim_gen_ts = torch.optim.Adam(generator_ts.parameters(), lr=learning_rate, betas=(beta1, beta2))

    discriminator_gt = Discriminator()
    discriminator_gt.cuda()
    discriminator_gt.train()
    optim_dis_gt = torch.optim.Adam(discriminator_gt.parameters(), lr=learning_rate, betas=(beta1, beta2))

    discriminator_gs = Discriminator()
    discriminator_gs.cuda()
    discriminator_gs.train()
    optim_dis_gs = torch.optim.Adam(discriminator_gs.parameters(), lr=learning_rate, betas=(beta1, beta2))

    discriminator_glot = GlobalDiscriminator()
    discriminator_glot.cuda()
    discriminator_glot.train()
    optim_dis_glot = torch.optim.Adam(discriminator_glot.parameters(), lr=learning_rate, betas=(beta1, beta2))

    discriminator_glos = GlobalDiscriminator()
    discriminator_glos.cuda()
    discriminator_glos.train()
    optim_dis_glos = torch.optim.Adam(discriminator_glos.parameters(), lr=learning_rate, betas=(beta1, beta2))

    clipper = RhoClipper(0, 1)

    iteration = 0

    for epoch in range(epochs):
        dataloader = DataLoader(dataset,
                                batch_size=batchsize,
                                shuffle=True,
                                collate_fn=collator,
                                drop_last=True)
        progress_bar = tqdm(dataloader)

        for i, data in enumerate(progress_bar):
            iteration += 1
            s, t = data

            fake_t, _, _ = generator_st(s)
            fake_s, _, _ = generator_ts(t)

            real_gs, real_gs_logit, _ = discriminator_gs(s)
            real_gt, real_gt_logit, _ = discriminator_gt(t)
            fake_gs, fake_gs_logit, _ = discriminator_gs(fake_s)
            fake_gt, fake_gt_logit, _ = discriminator_gt(fake_t)
            real_glos, real_glos_logit, _ = discriminator_glos(s)
            real_glot, real_glot_logit, _ = discriminator_glot(t)
            fake_glos, fake_glos_logit, _ = discriminator_glos(fake_s)
            fake_glot, fake_glot_logit, _ = discriminator_glot(fake_t)

            loss = discriminator_loss(fake_gt, real_gt, fake_gt_logit, real_gt_logit)
            loss += discriminator_loss(fake_gs, real_gs, fake_gs_logit, real_gs_logit)
            loss += discriminator_loss(fake_glos, real_glos, fake_glos_logit, real_glos_logit)
            loss += discriminator_loss(fake_glot, real_glot, fake_glot_logit, real_glot_logit)

            optim_dis_gs.zero_grad()
            optim_dis_gt.zero_grad()
            optim_dis_glos.zero_grad()
            optim_dis_glot.zero_grad()
            loss.backward()
            optim_dis_gs.step()
            optim_dis_gt.step()
            optim_dis_glos.step()
            optim_dis_glot.step()

            fake_t, fake_gen_t_logit, _ = generator_st(s)
            fake_s, fake_gen_s_logit, _ = generator_ts(t)

            fake_sts, _, _ = generator_ts(fake_t)
            fake_tst, _, _ = generator_st(fake_s)

            fake_t_id, fake_t_id_logit, _ = generator_st(t)
            fake_s_id, fake_s_id_logit, _ = generator_ts(s)

            fake_gs, fake_gs_logit, _ = discriminator_gs(fake_s)
            fake_gt, fake_gt_logit, _ = discriminator_gt(fake_t)
            fake_glot, fake_glot_logit, _ = discriminator_glot(fake_t)
            fake_glos, fake_glos_logit, _ = discriminator_glos(fake_s)

            loss = generator_loss(fake_gs, s, fake_gs_logit, fake_gen_s_logit, fake_sts, fake_s_id, fake_s_id_logit)
            loss += generator_loss(fake_gt, t, fake_gt_logit, fake_gen_t_logit, fake_tst, fake_t_id, fake_t_id_logit)
            loss += mseloss(fake_glot, torch.ones_like(fake_glot))
            loss += mseloss(fake_glot_logit, torch.ones_like(fake_glot_logit))
            loss += mseloss(fake_glos, torch.ones_like(fake_glos))
            loss += mseloss(fake_glos_logit, torch.ones_like(fake_glos_logit))

            optim_gen_st.zero_grad()
            optim_gen_ts.zero_grad()
            loss.backward()
            optim_gen_st.step()
            optim_gen_ts.step()

            generator_st.apply(clipper)
            generator_ts.apply(clipper)

            if iteration % interval == 0:
                torch.save(generator_st.state_dict(), f"{modeldir}/model_st_{iteration}.pt")
                torch.save(generator_ts.state_dict(), f"{modeldir}/model_ts_{iteration}.pt")

            print(f"iteration: {iteration} Loss: {loss.data}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UGATIT")
    parser.add_argument('--e', default=1000, type=int, help="the number of epochs")
    parser.add_argument('--i', default=1000, type=int, help="the interval of snapshot")
    parser.add_argument('--b', default=8, type=int, help="batch size")
    parser.add_argument('--modeldir', default="modeldir", type=Path, help="model output directory")
    parser.add_argument('--size', default=128, type=int, help="the size of training images")
    parser.add_argument('--lr', default=0.0001, type=float, help="learning rate of Adam")
    parser.add_argument('--b1', default=0.5, type=float, help="beta1 of Adam")
    parser.add_argument('--b2', default=0.999, type=float, help="beta2 of Adam")
    parser.add_argument('--src_path', type=Path, help="path which contains source images")
    parser.add_argument('--tgt_path', type=Path, help="path which contains target images")
    args = parser.parse_args()

    modeldir = args.modeldir
    modeldir.mkdir(exist_ok=True)

    train(args.e, args.b, args.i, modeldir, args.size, args.src_path, args.tgt_path, args.lr, args.b1, args.b2)
