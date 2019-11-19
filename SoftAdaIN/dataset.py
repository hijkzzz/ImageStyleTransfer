import os
import torch
import numpy as np
import cv2 as cv

from torch.utils.data import Dataset
from pathlib import Path


class CRDataset(Dataset):
    def __init__(self, coco_path: Path, cell_path: Path):
        super(CRDataset, self).__init__()

        self.mpath = coco_path
        self.mlist = list(self.mpath.glob("*.jpg"))
        self.mlen = len(self.mlist)
        self.tpath = cell_path
        self.tlist = list(self.tpath.glob("**/*.png"))
        self.tlen = len(self.tlist)

    def __len__(self):
        return self.mlen - 50

    def __getitem__(self, idx):
        m_path = self.mlist[idx]

        rnd = np.random.randint(self.tlen)
        t_path = self.tlist[rnd]

        return (m_path, t_path)


class CRTestDataset(Dataset):
    def __init__(self, coco_path: Path, cell_path: Path):
        super(CRTestDataset, self).__init__()

        self.mpath = coco_path
        self.mlist = list(self.mpath.glob("*.jpg"))
        self.mlen = len(self.mlist)
        self.tpath = cell_path
        self.tlist = list(self.tpath.glob("**/*.png"))
        self.tlen = len(self.tlist)

    def __len__(self):
        return 99

    def __getitem__(self, idx):
        m_path = self.mlist[idx]

        rnd = np.random.randint(self.tlen)
        t_path = self.tlist[rnd]

        return (m_path, t_path)


class CollateFn():
    def __init__(self):
        pass

    def _random_crop(self, img, size=384):
        height, width = img.shape[0], img.shape[1]

        if height > width:
            scale = size / width
        else:
            scale = size / height

        new_height, new_width = int(height * scale), int(width * scale)
        img = cv.resize(img, (new_width, new_height))

        rnd1 = np.random.randint(new_height - 224)
        rnd2 = np.random.randint(new_width - 224)

        img = img[rnd1: rnd1+224, rnd2: rnd2+224]

        return img

    def _crop(self, img, size=384):
        height, width = img.shape[0], img.shape[1]

        if height > width:
            scale = (size + 1) / width
        else:
            scale = (size + 1) / height

        new_height, new_width = int(height * scale), int(width * scale)
        img = cv.resize(img, (new_width, new_height))
        img = img[:384, :384, :]

        return img

    def _prepare(self, path, size=128):
        img = cv.imread(str(path))
        img = self._random_crop(img)
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def _prepare_test(self, path):
        img = cv.imread(str(path))
        img = self._crop(img)
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def __call__(self, batch):
        x_box = []
        y_box = []

        for b in batch:
            x_path, y_path = b
            x = self._prepare(x_path)
            y = self._prepare(y_path)
            x_box.append(x)
            y_box.append(y)

        x = torch.FloatTensor(np.array(x_box).astype(np.float32))
        y = torch.FloatTensor(np.array(y_box).astype(np.float32))

        x = x.cuda()
        y = y.cuda()

        return (x, y)

    def test(self, batch):
        x_box = []
        y_box = []

        for b in batch:
            x_path, y_path = b
            x = self._prepare_test(x_path)
            y = self._prepare_test(y_path)
            x_box.append(x)
            y_box.append(y)

        x = torch.FloatTensor(np.array(x_box).astype(np.float32))
        y = torch.FloatTensor(np.array(y_box).astype(np.float32))

        x = x.cuda()
        y = y.cuda()

        return (x, y)


class CollateFnTest():
    def __init__(self):
        pass

    def _crop(self, img, size=384):
        height, width = img.shape[0], img.shape[1]

        if height > width:
            scale = (size + 1) / width
        else:
            scale = (size + 1) / height

        new_height, new_width = int(height * scale), int(width * scale)
        img = cv.resize(img, (new_width, new_height))
        img = img[:384, :384, :]

        return img

    def _prepare_test(self, path):
        img = cv.imread(str(path))
        img = self._crop(img)
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def __call__(self, batch):
        x_box = []
        y_box = []

        for b in batch:
            x_path, y_path = b
            x = self._prepare_test(x_path)
            y = self._prepare_test(y_path)
            x_box.append(x)
            y_box.append(y)

        x = torch.FloatTensor(np.array(x_box).astype(np.float32))
        y = torch.FloatTensor(np.array(y_box).astype(np.float32))

        x = x.cuda()
        y = y.cuda()

        return (x, y)
