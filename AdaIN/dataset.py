import chainer
import cv2 as cv
import numpy as np

from chainer import cuda
from pathlib import Path

xp = cuda.cupy


class DatasetLoader:
    def __init__(self,
                 content_path: Path,
                 style_path: Path,
                 extension=".jpg",
                 coord_size=512,
                 crop_size=512):

        self.extension = extension
        self.coord_size = coord_size
        self.crop_size = crop_size

        self.content_path = content_path
        self.style_path = style_path

        self.content_train_list, self.content_val_list = self._train_val_split(self.content_path)
        self.style_train_list, self.style_val_list = self._train_val_split(self.style_path)

        self.content_len = len(self.content_train_list)
        self.style_len = len(self.style_train_list)

    def __repr__(self):
        return f"content length: {self.content_len} style length: {self.style_len}"

    @staticmethod
    def _variable(array_list):
        return chainer.as_variable(xp.array(array_list).astype(xp.float32))

    def _train_val_split(self, path: Path):
        image_list = list(path.glob(f"*{self.extension}"))
        all_len = len(image_list)

        split_point = int(all_len * 0.95)
        train_list = image_list[:split_point]
        val_list = image_list[split_point:]

        return train_list, val_list

    def _crop_content(self, img_array):
        height, width = img_array.shape[0:2]

        if height > width:
            scale = (self.coord_size + 1) / width
        else:
            scale = (self.coord_size + 1) / height

        new_width = int(width * scale)
        new_height = int(height * scale)
        resized_img = cv.resize(img_array, (new_width, new_height))

        up = np.random.randint(new_height - self.crop_size)
        left = np.random.randint(new_width - self.crop_size)

        cropped_img = resized_img[up: up + self.crop_size, left: left + self.crop_size]

        return cropped_img

    def _crop_style(self, img_array):
        up = np.random.randint(self.coord_size - self.crop_size + 1)
        left = np.random.randint(self.coord_size - self.crop_size + 1)

        cropped_img = img_array[up: up + self.crop_size, left: left + self.crop_size]

        return cropped_img

    def _prepare(self, path: Path, img_type="content"):
        img_array = cv.imread(str(path))

        if img_type == "content":
            cropped = self._crop_content(img_array)
        elif img_type == "style":
            cropped = self._crop_style(img_array)
        else:
            raise AttributeError

        img = cropped[:, :, ::-1]
        img = (img.transpose(2, 0, 1) - 127.5) / 127.5

        return img

    def _get_train_img(self, mode="content"):
        if mode == "content":
            rnd = np.random.randint(self.content_len)
            path = self.content_train_list[rnd]
        else:
            rnd = np.random.randint(self.style_len)
            path = self.style_train_list[rnd]

        img = self._prepare(path, mode)

        return img

    def _get_valid_img(self, index, mode="content"):
        if mode == "content":
            path = self.content_val_list[index]
        else:
            path = self.style_val_list[index]

        img = self._prepare(path, mode)

        return img

    def train(self, batchsize):
        con_box = []
        sty_box = []

        for _ in range(batchsize):
            content = self._get_train_img(mode="content")
            style = self._get_train_img(mode="style")

            con_box.append(content)
            sty_box.append(style)

        content = self._variable(con_box)
        style = self._variable(sty_box)

        return content, style

    def valid(self, validsize):
        con_box = []
        sty_box = []

        for index in range(validsize):
            content = self._get_valid_img(index, mode="content")
            style = self._get_valid_img(index, mode="style")

            con_box.append(content)
            sty_box.append(style)

        content = self._variable(con_box)
        style = self._variable(sty_box)

        return content, style
