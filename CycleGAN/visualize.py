import numpy as np

import matplotlib
matplotlib.use('Agg')
import pylab


class Visualization:
    def __init__(self):
        pass

    @staticmethod
    def _coordinate(array):
        tmp = np.clip(array * 127.5 + 127.5, 0, 255).transpose(2, 0, 1).astype(np.uint8)

        return tmp

    def __call__(self, src, tgt, outdir, epoch, testsize):
        height = int(testsize / 2)

        pylab.rcParams['figure.figsize'] = (16.0, 16.0)
        pylab.clf()

        for index in range(testsize):
            tmp = np.clip(src[index]*127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)
            pylab.subplot(height, 4, 2 * index + 1)
            pylab.imshow(tmp)
            pylab.axis('off')
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")
            tmp = np.clip(tgt[index]*127.5 + 127.5, 0, 255).transpose(1, 2, 0).astype(np.uint8)
            pylab.subplot(height, 4, 2 * index + 2)
            pylab.imshow(tmp)
            pylab.axis('off')
            pylab.savefig(f"{outdir}/visualize_{epoch}.png")
