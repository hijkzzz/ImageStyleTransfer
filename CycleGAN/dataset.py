import chainer
import cv2 as cv
import numpy as np

from pathlib import Path
from chainer import cuda

from sklearn.model_selection import train_test_split

xp = cuda.cupy
cuda.get_device(0).use()


class DatasetLoader:
    def __init__(self,
                 src_path: Path,
                 tgt_path: Path,
                 extension=".png",
                 size=128):

        self.src_path = src_path
        self.tgt_path = tgt_path

        self.extension = extension
        self.size = size

        self.src_list, self.src_len = self._glob_and_length(src_path)
        self.tgt_list, self.tgt_len = self._glob_and_length(tgt_path)

        self.src_train_list, self.src_val_list = self._train_val_split()
        self.src_len = len(self.src_train_list)

    def __repr__(self):
        return f"source dataset length: {self.src_len} target dataset length: {self.tgt_len}"

    def _glob_and_length(self, path: Path):
        pathlist = list(path.glob(f"*{self.extension}"))
        pathlen = len(pathlist)

        return pathlist, pathlen

    def _train_val_split(self):
        split_point = int(len(self.src_list) * 0.95)
        x_train = self.src_list[:split_point]
        x_val = self.src_list[split_point:]

        return x_train, x_val

    @staticmethod
    def _vairble(array_list):
        return chainer.as_variable(xp.array(array_list).astype(xp.float32))

    def _prepare_img(self, path: Path):
        img = cv.imread(str(path))
        img = cv.resize(img, (self.size, self.size), interpolation=cv.INTER_CUBIC)

        # horizontal flip
        if np.random.randint(2):
            img = img[:, ::-1, :]

        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def _get_img(self, mode="source"):
        if mode == "source":
            rnd = np.random.randint(self.src_len)
            img_path = self.src_train_list[rnd]

        elif mode == "target":
            rnd = np.random.randint(self.tgt_len)
            img_path = self.tgt_list[rnd]

        img = self._prepare_img(img_path)

        return img

    def train(self, batchsize):
        src_box = []
        tgt_box = []

        for _ in range(batchsize):
            src = self._get_img(mode="source")
            tgt = self._get_img(mode="target")

            src_box.append(src)
            tgt_box.append(tgt)

        src = self._vairble(src_box)
        tgt = self._vairble(tgt_box)

        return src, tgt

    def valid(self, validsize):
        src_box = []

        for index in range(validsize):
            img_path = self.src_val_list[index]
            img = self._prepare_img(img_path)
            src_box.append(img)

        src = self._vairble(src_box)

        return src


class TestDatasetLoader:
    def __init__(self,
                 src_path: Path,
                 extension=".png",
                 size=128):

        self.src_path = src_path

        self.extension = extension
        self.size = size

        self.src_list, self.src_len = self._glob_and_length(src_path)

    def __repr__(self):
        return f"source dataset length: {self.src_len}"

    def _glob_and_length(self, path: Path):
        pathlist = list(path.glob(f"*{self.extension}"))
        pathlen = len(pathlist)

        return pathlist, pathlen

    @staticmethod
    def _vairble(array_list):
        return chainer.as_variable(xp.array(array_list).astype(xp.float32))

    def _prepare_img(self, path: Path):
        img = cv.imread(str(path))
        img = cv.resize(img, (self.size, self.size), interpolation=cv.INTER_CUBIC)

        # horizontal flip
        if np.random.randint(2):
            img = img[:, ::-1, :]

        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def test(self):
        src_box = []

        for index in range(self.src_len):
            img_path = self.src_list[index]
            img = self._prepare_img(img_path)
            src_box.append(img)

        src = self._vairble(src_box)

        return src
