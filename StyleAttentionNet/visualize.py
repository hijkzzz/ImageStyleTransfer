import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab


class Visualizer:
    def __init__(self):
        pass

    def _convert(self, img_array):
        tmp = np.clip(img_array*127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)

        return tmp

    def __call__(self, con, sty, y, outdir, number, validsize):
        pylab.rcParams['figure.figsize'] = (16.0,16.0)
        pylab.clf()

        for index in range(validsize):
            tmp = self._convert(con[index])
            pylab.subplot(validsize, validsize, validsize*index+1)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{number}.png")
            tmp = self._convert(sty[index])
            pylab.subplot(validsize, validsize, validsize*index+2)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{number}.png")
            tmp = self._convert(y[index])
            pylab.subplot(validsize, validsize, validsize*index+3)
            pylab.imshow(tmp)
            pylab.axis("off")
            pylab.savefig(f"{outdir}/visualize_{number}.png")
