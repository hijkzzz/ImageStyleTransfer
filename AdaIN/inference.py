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


def infer(testsize,
          outdir,
          model_path,
          con_path,
          sty_path,
          extension,
          coord_size,
          crop_size,
          alpha,
          ):
    
    # Dataset definition
    dataloader = DatasetLoader(con_path, sty_path, extension, coord_size, crop_size)
    print(dataloader)
    con_valid, sty_valid = dataloader.valid(testsize)

    # Mode & Optimizer defnition
    decoder = Decoder()
    decoder.to_gpu()
    serializers.load_npz(model_path, decoder)

    vgg = VGG()
    vgg.to_gpu()

    # Visualizer definition
    visualizer = Visualizer()

    with chainer.using_config("train", False):
        style_feat_list = vgg(sty_valid)
        content_feat = vgg(con_valid)[-1]

        t = adain(content_feat, style_feat_list[-1])
        t = alpha * t + (1 - alpha) * content_feat
        g_t = decoder(t)

    g_t = g_t.data.get()
    con = con_valid.data.get()
    sty = sty_valid.data.get()

    visualizer(con, sty, g_t, outdir, 0, testsize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AdaIN")
    parser.add_argument("--t", type=int, default=12, help="test size")
    parser.add_argument("--outdir", type=Path, default="inferdir", help="output directory")
    parser.add_argument("--ext", type=str, default=".jpg", help="extension of training image")
    parser.add_argument("--coord", type=int, default=512, help="the coordinate size of image")
    parser.add_argument("--crop", type=int, default=512, help="the cropped size of image")
    parser.add_argument("--alpha", type=float, default=1.0, help="alpha of AdaIN")
    parser.add_argument("--con_path", type=Path, help="path containing contain images")
    parser.add_argument("--sty_path", type=Path, help="path containing style images")
    parser.add_argument("--model", type=Path, help="trained model path")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    infer(args.t, outdir, args.model, args.con_path, args.sty_path, args.ext,
          args.coord, args.crop, args.alpha)
