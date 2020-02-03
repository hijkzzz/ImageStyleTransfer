import torch
import torch.nn as nn
import numpy as np
import argparse

from pathlib import Path
from tqdm import tqdm
from model import Model
from dataset import CSDataset, ImageCollate
from visualize import Visualizer
from torch.utils.data import DataLoader


def infer(content_path,
          style_path,
          model_path,
          testsize,
          outdir):

    # Dataset definition
    dataset = CSDataset(c_path=content_path, s_path=style_path, mode="test")
    collator = ImageCollate(test=True)

    # Model definition
    model = Model()
    model.load_state_dict(torch.load(model_path))
    model.cuda()
    model.eval()

    # Visualizer definition
    visualizer = Visualizer()

    dataloader = DataLoader(dataset,
                            batch_size=testsize,
                            shuffle=True,
                            collate_fn=collator,
                            drop_last=True)
    progress_bar = tqdm(dataloader)

    for index, data in enumerate(progress_bar):
        c, s = data

        with torch.no_grad():
            _, _, _, _, y = model(c, s)

        y = y.detach().cpu().numpy()
        c = c.detach().cpu().numpy()
        s = s.detach().cpu().numpy()

        visualizer(c, s, y, outdir, index, testsize)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='StyleAttentionNetwork')
    parser.add_argument('--t', type=int, default=3, help="batch size in inference phase")
    parser.add_argument('--outdir', type=Path, default='inferdir', help="result directory")
    parser.add_argument('--con_path', type=Path, help="path containing content images")
    parser.add_argument('--sty_path', type=Path, help="path containing style images")
    parser.add_argument('--model', type=Path, help="trained model path")
    args = parser.parse_args()

    outdir = args.outdir
    outdir.mkdir(exist_ok=True)

    infer(args.con_path, args.sty_path, args.model, args.t, outdir)
