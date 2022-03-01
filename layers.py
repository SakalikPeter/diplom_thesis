import tensorflow as tf
from tensorflow.keras.layers import Layer


class PixelNorm(Layer):
    def __init__(self, epsilon=1e-8):
        super(PixelNorm, self).__init__()
        self.epsilon = epsilon

    def call(self, input_tensor, **kwargs):
        return input_tensor / tf.math.sqrt(tf.reduce_mean(input_tensor ** 2, axis=-1, keepdims=True) + self.epsilon)


class MinibatchStd(Layer):
    def __init__(self, group_size=4, epsilon=1e-8):
        super(MinibatchStd, self).__init__()
        self.epsilon = epsilon
        self.group_size = group_size

    def call(self, input_tensor, **kwargs):
        n, h, w, c = input_tensor.shape
        x = tf.reshape(input_tensor, [self.group_size, -1, h, w, c])
        group_mean, group_var = tf.nn.moments(x, axes=0, keepdims=False)
        group_std = tf.sqrt(group_var + self.epsilon)
        avg_std = tf.reduce_mean(group_std, axis=[1, 2, 3], keepdims=True)
        x = tf.tile(avg_std, [self.group_size, h, w, 1])

        return tf.concat([input_tensor, x], axis=-1)


class FadeIn(Layer):
    @tf.function
    def call(self, input_alpha, a, b):
        alpha = tf.reduce_mean(input_alpha)
        y = alpha * a + (1. - alpha) * b
        return y


class Conv2D(Layer):
    def __init__(self, out_channels, kernel=3, gain=2, **kwargs):
        super(Conv2D, self).__init__(kwargs)
        self.kernel = kernel
        self.out_channels = out_channels
        self.gain = gain
        self.pad = kernel != 1
        self.in_channels = None
        self.w = None
        self.b = None
        self.scale = None

    def build(self, input_shape):
        self.in_channels = input_shape[-1]

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.w = self.add_weight(shape=[self.kernel,
                                        self.kernel,
                                        self.in_channels,
                                        self.out_channels],
                                 initializer=initializer,
                                 trainable=True, name='kernel')

        self.b = self.add_weight(shape=(self.out_channels,),
                                 initializer='zeros',
                                 trainable=True, name='bias')

        fan_in = self.kernel * self.kernel * self.in_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    def call(self, inputs, **kwargs):
        if self.pad:
            x = tf.pad(inputs, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        else:
            x = inputs
        output = tf.nn.conv2d(x, self.scale * self.w, strides=1, padding="VALID") + self.b
        return output


class Dense(Layer):
    def __init__(self, units, gain=2, lrmul=1, **kwargs):
        super(Dense, self).__init__(kwargs)
        self.units = units
        self.gain = gain
        self.lrmul = lrmul
        self.in_channels = None
        self.w = None
        self.b = None
        self.scale = None

    def build(self, input_shape):
        self.in_channels = input_shape[-1]
        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1. / self.lrmul)
        self.w = self.add_weight(shape=[self.in_channels,
                                        self.units],
                                 initializer=initializer,
                                 trainable=True, name='kernel')

        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True, name='bias')

        fan_in = self.in_channels
        self.scale = tf.sqrt(self.gain / fan_in)

    @tf.function
    def call(self, inputs):
        output = tf.matmul(inputs, self.scale * self.w) + self.b
        return output * self.lrmul


class AddNoise(Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.B = None

    def build(self, input_shape):
        n, h, w, c = input_shape[0]

        initializer = tf.keras.initializers.RandomNormal(mean=0., stddev=1.)
        self.B = self.add_weight(shape=[1, 1, 1, c],
                                 initializer=initializer,
                                 trainable=True, name='kernel')

    def call(self, inputs, **kwargs):
        x, noise = inputs
        output = x + self.B * noise
        return output


class AdaIN(Layer):
    def __init__(self, gain=1, **kwargs):
        super(AdaIN, self).__init__(kwargs)
        self.gain = gain
        self.w_channels = None
        self.x_channels = None
        self.dense_1 = None
        self.dense_2 = None

    def build(self, input_shapes):
        x_shape = input_shapes[0]
        w_shape = input_shapes[1]

        self.w_channels = w_shape[-1]
        self.x_channels = x_shape[-1]

        self.dense_1 = Dense(self.x_channels, gain=1)
        self.dense_2 = Dense(self.x_channels, gain=1)

    def call(self, inputs, **kwargs):
        x, w = inputs
        ys = tf.reshape(self.dense_1(w), (-1, 1, 1, self.x_channels))
        yb = tf.reshape(self.dense_2(w), (-1, 1, 1, self.x_channels))

        output = ys * x + yb
        return output
