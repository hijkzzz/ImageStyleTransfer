import chainer
import numpy as np
import copy
import cv2 as cv
import pickle

from pathlib import Path
from chainer import cuda

xp = cuda.cupy
cuda.get_device(0).use()


class DatasetLoader:
    def __init__(self,
                 data_path: Path,
                 extension='.png',
                 img_size=128
                 ):

        self.data_path = data_path
        self.belong_list = [path.name for path in self.data_path.iterdir()]
        self.belong_hashmap = self._hashmap(self.belong_list)
        self.number = len(self.belong_list)
        self.extension = extension
        self.size = img_size

        print(self.belong_list)
        print(self.belong_hashmap)

    @staticmethod
    def _hashmap(belong_list):
        belong_len = len(belong_list)
        hashmap = {}
        for belong, index in zip(belong_list, range(belong_len)):
            hashmap[belong] = index

        return hashmap

    @staticmethod
    def _label_remove(label_list, source):
        label_list.remove(source)

        return label_list

    @staticmethod
    def _variable(array_list, array_type='float'):
        if array_type == 'float':
            return chainer.as_variable(xp.array(array_list).astype(xp.float32))

        else:
            return chainer.as_variable(xp.array(array_list).astype(xp.int32))

    def _onehot_convert(self, label, path):
        onehot = np.zeros(self.number)
        onehot[self.belong_hashmap[label]] = 1

        return onehot

    def _prepare(self, path: Path):
        img = cv.imread(str(path))
        img = cv.resize(img, (self.size, self.size))
        img = img[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def _get_path_onehot(self, label_list):
        rnd_belong = np.random.choice(label_list)
        pathlist = list((self.data_path / Path(str(rnd_belong))).glob(f"*{self.extension}"))

        img_path = np.random.choice(pathlist)
        onehot = self._onehot_convert(rnd_belong, img_path)
        img = self._prepare(img_path)

        return rnd_belong, img, onehot

    def train(self, batchsize):
        x_img_box = []
        x_label_box = []
        y_img_box = []
        y_label_box = []
        z_img_box = []
        z_label_box = []
        for _ in range(batchsize):
            belong_list = copy.copy(self.belong_list)
            rnd_belong, img, onehot = self._get_path_onehot(belong_list)

            x_img_box.append(img)
            x_label_box.append(onehot)

            belong_list = self._label_remove(belong_list, rnd_belong)
            rnd_belong, img, onehot = self._get_path_onehot(belong_list)

            y_img_box.append(img)
            y_label_box.append(onehot)

            belong_list = self._label_remove(belong_list, rnd_belong)
            rnd_belong, img, onehot = self._get_path_onehot(belong_list)

            z_img_box.append(img)
            z_label_box.append(onehot)

        x_sp = self._variable(x_img_box)
        x_label = self._variable(x_label_box, array_type='float')
        y_sp = self._variable(y_img_box)
        y_label = self._variable(y_label_box, array_type='float')
        z_sp = self._variable(z_img_box)
        z_label = self._variable(z_label_box, array_type='float')

        return (x_sp, x_label, y_sp, y_label, z_sp, z_label)

    def test(self, testsize):
        testlist = list(self.data_path.glob("*.png"))
        test_box = []

        for path in testlist:
            img = self._prepare(path)
            test_box.append(img)

        x_test = self._variable(test_box)

        return x_test
