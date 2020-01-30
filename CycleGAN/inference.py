import chainer
import chainer.functions as F
import argparse
import numpy as np

from pathlib import Path
from chainer import serializers, cuda
from model import Generator, Discriminator
from utils import set_optimizer
from dataset import TestDatasetLoader
from visualize import Visualization

xp = cuda.cupy
cuda.get_device(0).use()


def infer(testsize,
          inferdir,
          model_path,
          src_path,
          extension,
          img_size
          ):

    # Dataset definition
    dataloader = TestDatasetLoader(src_path, extension, img_size)
    print(dataloader)
    src_val = dataloader.test()

    # Model & Optimizer definition
    generator_xy = Generator()
    generator_xy.to_gpu()
    serializers.load_npz(model_path, generator_xy)

    # Visualization
    visualizer = Visualization()

    with chainer.using_config('train', False):
        tgt = generator_xy(src_val)

    src = src_val.data.get()
    tgt = tgt.data.get()

    visualizer(src, tgt, inferdir, 0, testsize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CycleGAN")
    parser.add_argument("--v", type=int, default=8, help="valid size")
    parser.add_argument("--inferdir", type=Path, default="inferdir", help="output directory")
    parser.add_argument("--model", type=Path, help="trained model path")
    parser.add_argument('--ext', type=str, default=".png", help="the extension of training data")
    parser.add_argument("--size", type=int, default=128, help="the size of training data")
    parser.add_argument("--src_path", type=Path, help="path which contains source data")
    args = parser.parse_args()

    inferdir = args.inferdir
    inferdir.mkdir(exist_ok=True)

    infer(args.v, args.inferdir, args.model, args.src_path, args.ext, args.size)
