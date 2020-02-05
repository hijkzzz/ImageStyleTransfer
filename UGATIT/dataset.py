import numpy as np
import cv2 as cv
import torch

from torch.utils.data import Dataset
from pathlib import Path


class UGATITDataset(Dataset):
    def __init__(self, src_path: Path, tgt_path: Path):
        self.src_path = src_path
        self.tgt_path = tgt_path

        self.src_list = list(src_path.glob('*.jpg'))
        exts = ['.jpg', '.png']
        self.tgt_list = [img for img in self.tgt_path.glob('**/*') if img.suffix in exts]

        self.src_len = len(self.src_list)
        self.tgt_len = len(self.tgt_list)

    def __str__(self):
        return f"source dataset length: {self.src_len} target dataset length: {self.tgt_len}"

    def __len__(self):
        return self.src_len

    def __getitem__(self, index):
        src_path = self.src_list[index]

        rnd = np.random.randint(self.tgt_len)
        tgt_path = self.tgt_list[rnd]

        return src_path, tgt_path


class UGATITDatasetTest(Dataset):
    def __init__(self, src_path):
        self.src_path = src_path
        self.src_list = list(self.src_path.glob('*.png'))
        self.src_len = len(self.src_list)

    def __str__(self):
        return f"source dataset length: {self.src_len}"

    def __len__(self):
        return 50

    def __getitem__(self, index):
        src_path = self.src_list[index]

        return src_path


class ImageCollate:
    def __init__(self, size=128):
        self.size = size

    @staticmethod
    def _totensor(array_list):
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    def _prepare(self, path: Path):
        img = cv.imread(str(path))
        img = cv.resize(img, (self.size, self.size), interpolation=cv.INTER_CUBIC)
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def __call__(self, batch):
        src_box = []
        tgt_box = []

        for b in batch:
            src_path, tgt_path = b
            src = self._prepare(src_path)
            tgt = self._prepare(tgt_path)

            src_box.append(src)
            tgt_box.append(tgt)

        src = self._totensor(src_box)
        tgt = self._totensor(tgt_box)

        return src, tgt


class ImageCollateTest:
    def __init__(self, size=128):
        self.size = size

    @staticmethod
    def _totensor(array_list):
        return torch.cuda.FloatTensor(np.array(array_list).astype(np.float32))

    def _prepare(self, path: Path):
        img = cv.imread(str(path))
        img = cv.resize(img, (self.size, self.size), interpolation=cv.INTER_CUBIC)
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def __call__(self, batch):
        src_box = []

        for b in batch:
            src_path = b
            src = self._prepare(src_path)
            src_box.append(src)

        src = self._totensor(src_box)

        return src
