import torch
import numpy as np
import cv2 as cv

from torch.utils.data import Dataset
from pathlib import Path


class CSDataset(Dataset):
    def __init__(self,
                 c_path: Path,
                 s_path: Path,
                 extension='.jpg',
                 mode='train'
                 ):

        self.extension = extension
        self.mode = mode

        self.c_list, self.c_len = self._get_list_length(c_path)
        self.s_list, self.s_len = self._get_list_length(s_path)

    def __repr__(self):
        return f"content length: {self.c_len} style length: {self.s_len}"

    def _get_list_length(self, path: Path):
        pathlist = list(path.glob(f"*{self.extension}"))
        pathlen = len(pathlist)

        return pathlist, pathlen

    def __len__(self):
        if self.mode == 'train':
            return self.c_len
        else:
            return 99

    def __getitem__(self, index):
        c_path = self.c_list[index]

        rnd = np.random.randint(self.s_len)
        s_path = self.s_list[rnd]

        return (c_path, s_path)


class ImageCollate():
    def __init__(self, test=False):
        self.test = test

    def _preapre(self, filename):
        image_path = filename
        image = cv.imread(str(image_path))
        if not image is None:
            height, width = image.shape[0], image.shape[1]

            if height > width:
                scale = 513 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv.resize(image, (new_width, new_height))
            
            if height <= width:
                scale = 513 / height
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv.resize(image, (new_width, new_height))

            rnd1 = np.random.randint(200)
            rnd2 = np.random.randint(200)
            image = image[rnd1: rnd1 + 256, rnd2: rnd2 + 256]

            height, width = image.shape[0], image.shape[1]

            image = image[:, :, ::-1]
            image = image.transpose(2, 0, 1)
            image = (image - 127.5)/127.5

            return image

    def _test_preapre(self, filename):
        image_path = filename
        image = cv.imread(str(image_path))
        if not image is None:
            height, width = image.shape[0], image.shape[1]

            if height > width:
                scale = 513 / width
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv.resize(image, (new_width, new_height))
            
            if height <= width:
                scale = 513 / height
                new_width = int(width * scale)
                new_height = int(height * scale)
                image = cv.resize(image, (new_width, new_height))

            #rnd1 = np.random.randint(100)
            #rnd2 = np.random.randint(100)
            image = image[:512, :512]

            height, width = image.shape[0], image.shape[1]

            image = image[:, :, ::-1]
            image = image.transpose(2, 0, 1)
            image = (image - 127.5)/127.5

            return image

    def __call__(self, batch):
        c_box = []
        s_box = []
        for b in batch:
            c_name, s_name = b
            if self.test:
                s = self._test_preapre(s_name)
                c = self._test_preapre(c_name)
            
            else:
                s = self._preapre(s_name)
                c = self._preapre(c_name)

            c_box.append(c)
            s_box.append(s)

        c = np.array(c_box).astype(np.float32)
        s = np.array(s_box).astype(np.float32)

        c = torch.FloatTensor(c)
        s = torch.FloatTensor(s)

        c = c.cuda()
        s = s.cuda()

        return (c, s)
