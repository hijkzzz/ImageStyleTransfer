from chainer import optimizers


def set_optimizer(model, alpha, beta1, beta2):
    optimizer = optimizers.Adam(alpha=alpha, beta1=beta1, beta2=beta2)
    optimizer.setup(model)

    return optimizer
