import tensorflow as tf


def wasserstein_loss(y_true, y_pred):
    return -tf.reduce_mean(y_true * y_pred)


def binary_cross_entropy(fake, real):
    bce = tf.keras.losses.BinaryCrossentropy()
    return (bce(fake[0], fake[1]) + bce(real[0], real[1])) / 2


def categorical_cross_entropy(fake, real):
    cce = tf.keras.losses.CategoricalCrossentropy()
    return (cce(fake[0], fake[1]) + cce(real[0], real[1])) / 2
