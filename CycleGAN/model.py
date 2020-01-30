import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L

from chainer import cuda, Chain, initializers

xp = cuda.cupy
cuda.get_device(0).use()


class CBR(Chain):
	def __init__(self,
				 in_ch,
				 out_ch,
				 kernel,
				 stride,
				 padding,
				 activation=F.relu,
				 down=False,
				 up=False):

		w = initializers.Normal(0.02)
		super(CBR, self).__init__()

		self.activ = activation
		self.down = down
		self.up = up

		with self.init_scope():
			self.c = L.Convolution2D(in_ch, out_ch, kernel, stride, padding, initialW=w)
			self.bn = L.BatchNormalization(out_ch)

	def __call__(self, x):
		if self.up:
			x = F.unpooling_2d(x, 2, 2, 0, cover_all=False)

		h = self.activ(self.bn(self.c(x)))

		return h


class ResBlock(Chain):
	def __init__(self, in_ch, out_ch):
		w = initializers.Normal(0.02)
		super(ResBlock, self).__init__()
		with self.init_scope():
			self.c0 = L.Convolution2D(in_ch, out_ch, 3, 1, 1, initialW=w)
			self.c1 = L.Convolution2D(out_ch, out_ch, 3, 1, 1, initialW=w)

			self.bn0 = L.BatchNormalization(out_ch)
			self.bn1 = L.BatchNormalization(out_ch)

	def __call__(self, x):
		h = F.relu(self.bn0(self.c0(x)))
		h = F.relu(self.bn1(self.c1(h)))

		return h + x


class Generator(Chain):
	def __init__(self, base=32, res_layers=6):
		w = initializers.Normal(0.02)
		super(Generator, self).__init__()

		with self.init_scope():
			self.c0 = CBR(3, base, 7, 1, 3)
			self.c1 = CBR(base, base*2, 4, 2, 1, down=True)
			self.c2 = CBR(base*2, base*4, 4, 2, 1, down=True)

			self.res = chainer.ChainList()

			for _ in range(res_layers):
				self.res.add_link(ResBlock(base*4, base*4))

			self.c3 = CBR(base*4, base*2, 3, 1, 1, up=True)
			self.c4 = CBR(base*2, base, 3, 1, 1, up=True)
			self.c5 = L.Convolution2D(base, 3, 7, 1, 3, initialW=w)

	def __call__(self, x):
		h = self.c0(x)
		h = self.c1(h)
		h = self.c2(h)
		for link in self.res.children():
			h = link(h)
		h = self.c3(h)
		h = self.c4(h)
		h = self.c5(h)

		return F.tanh(h)


class Discriminator(Chain):
	def __init__(self, base=64):
		w = initializers.Normal(0.02)
		super(Discriminator, self).__init__()

		with self.init_scope():
			self.c0 = CBR(3, base, 4, 2, 1, down=True, activation=F.leaky_relu)
			self.c1 = CBR(base, base*2, 4, 2, 1, down=True, activation=F.leaky_relu)
			self.c2 = CBR(base*2, base*4, 4, 2, 1, down=True, activation=F.leaky_relu)
			self.c3 = CBR(base*4, base*8, 4, 2, 1, down=True, activation=F.leaky_relu)
			self.c4 = L.Convolution2D(base*8, 1, 1, 1, 0, initialW=w)

	def __call__(self, x):
		h = self.c0(x)
		h = self.c1(h)
		h = self.c2(h)
		h = self.c3(h)
		h = self.c4(h)

		return h
