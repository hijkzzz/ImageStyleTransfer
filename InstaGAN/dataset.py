import chainer
import cv2 as cv
import numpy as np

from pathlib import Path
from chainer import cuda

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

        self.src_list = list(self.src_path.glob(f"mask/*{extension}"))
        self.tgt_list = list(self.tgt_path.glob(f"mask/*{extension}"))

        self.src_train_list, self.src_val_list = self._train_val_split(self.src_list)
        self.tgt_train_list, self.tgt_val_list = self._train_val_split(self.tgt_list)

        self.src_len = len(self.src_train_list)
        self.tgt_len = len(self.tgt_train_list)

        self.extension = extension
        self.size = size

    def __repr__(self):
        return f"source dataset length: {self.src_len} target dataset length: {self.tgt_len}"

    def _train_val_split(self, pathlist):
        split_point = int(len(pathlist) * 0.95)
        train = pathlist[:split_point]
        val = pathlist[split_point:]

        return train, val

    @staticmethod
    def _variable(array_list):
        return chainer.as_variable(xp.array(array_list).astype(xp.float32))

    def _prepare_image_mask(self, img_path: Path, mask_path: Path):
        img = cv.imread(str(img_path))
        mask = cv.imread(str(mask_path))

        img = cv.resize(img, (self.size, self.size), interpolation=cv.INTER_CUBIC)
        mask = cv.resize(mask, (self.size, self.size), interpolation=cv.INTER_CUBIC)

        img = img[:, :, ::-1]
        mask = mask[:, :, ::-1]
        mask = mask[:, :, 2].reshape(self.size, self.size, 1)

        # horizontal flip
        if np.random.randint(2):
            img = img[:, ::-1, :]
            mask = mask[:, ::-1, :]

        img = (img.transpose(2, 0, 1) - 127.5) / 127.5
        mask = mask.transpose(2, 0, 1) / 255.0

        return img, mask

    def _get_img_train(self, mode="src"):
        if mode == "src":
            rnd = np.random.randint(self.src_len)
            mask_path = self.src_train_list[rnd]
            img_path = self.src_path / Path("image") / mask_path.name

        elif mode == "tgt":
            rnd = np.random.randint(self.tgt_len)
            mask_path = self.tgt_train_list[rnd]
            img_path = self.tgt_path / Path("image") / mask_path.name

        img, mask = self._prepare_image_mask(img_path, mask_path)

        return img, mask

    def _get_img_valid(self, index, mode="src"):
        if mode == "src":
            mask_path = self.src_val_list[index]
            img_path = self.src_path / Path("image") / mask_path.name

        elif mode == "tgt":
            mask_path = self.tgt_val_list[index]
            img_path = self.tgt_path / Path("image") / mask_path.name

        img, mask = self._prepare_image_mask(img_path, mask_path)

        return img, mask

    def train(self, batchsize):
        src_box = []
        src_mask_box = []
        tgt_box = []
        tgt_mask_box = []

        for _ in range(batchsize):
            src, src_mask = self._get_img_train(mode="src")
            tgt, tgt_mask = self._get_img_train(mode="tgt")

            src_box.append(src)
            tgt_box.append(tgt)
            src_mask_box.append(src_mask)
            tgt_mask_box.append(tgt_mask)

        src = self._variable(src_box)
        tgt = self._variable(tgt_box)
        src_mask = self._variable(src_mask_box)
        tgt_mask = self._variable(tgt_mask_box)

        return src, src_mask, tgt, tgt_mask

    def valid(self, validsize):
        src_box = []
        src_mask_box = []
        tgt_box = []
        tgt_mask_box = []

        for index  in range(validsize):
            src, src_mask = self._get_img_valid(index, mode="src")
            tgt, tgt_mask = self._get_img_valid(index, mode="tgt")

            src_box.append(src)
            tgt_box.append(tgt)
            src_mask_box.append(src_mask)
            tgt_mask_box.append(tgt_mask)

        src = self._variable(src_box)
        tgt = self._variable(tgt_box)
        src_mask = self._variable(src_mask_box)
        tgt_mask = self._variable(tgt_mask_box)

        return src, src_mask, tgt, tgt_mask
