from chainer import optimizers


def set_optimizer(model, alpha, beta):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta)
    optimizer.setup(model)

    return optimizer
