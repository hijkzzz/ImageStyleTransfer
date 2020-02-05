import numpy as np

import matplotlib
matplotlib.use("Agg")
import pylab


class Visualizer:
    def __init__(self):
        pass

    @staticmethod
    def _convert_img(img):
        return np.clip(img*127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

    @staticmethod
    def _convert_mask(mask):
        h = np.clip(mask*255.0, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return np.squeeze(h)

    def __call__(self, x, x_mask, y, y_mask, outdir, epoch, validsize, switch):
        pylab.rcParams['figure.figsize'] = (16.0,16.0)
        pylab.clf()

        for index in range(validsize):
            tmp = self._convert_img(x[index])
            pylab.subplot(validsize, validsize, validsize*index+1)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{switch}_{epoch}.png")
            tmp = self._convert_mask(x_mask[index])
            pylab.subplot(validsize, validsize, validsize*index+2)
            pylab.imshow(tmp, cmap='gray')
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{switch}_{epoch}.png")
            tmp = self._convert_img(y[index])
            pylab.subplot(validsize, validsize, validsize*index+3)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{switch}_{epoch}.png")
            tmp = self._convert_mask(y_mask[index])
            pylab.subplot(validsize, validsize, validsize*index+4)
            pylab.imshow(tmp, cmap='gray')
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{switch}_{epoch}.png")
