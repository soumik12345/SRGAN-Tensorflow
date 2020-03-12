import tensorflow as tf


class UpSample(tf.keras.layers.Layer):

    def __init__(self, filters):
        super(UpSample, self).__init__()
        self.convolution_layer = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.pixel_shuffle_layer = tf.keras.layers.Lambda(UpSample.pixel_shuffle(2))
        self.activation_layer = tf.keras.layers.PReLU(shared_axes=[1, 2])

    @staticmethod
    def pixel_shuffle(scale):
        return lambda x: tf.nn.depth_to_space(x, scale)

    def call(self, inputs):
        y = self.convolution_layer(inputs)
        y = self.pixel_shuffle_layer(y)
        y = self.activation_layer(y)
        return y


class Residual(tf.keras.layers.Layer):

    def __init__(self, filters, momentum=0.8):
        super(Residual, self).__init__()
        self.convolution_1 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.batch_norm_1 = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.activation_1 = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.convolution_2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.batch_norm_2 = tf.keras.layers.BatchNormalization(momentum=momentum)
        self.addition = tf.keras.layers.Add()

    def call(self, inputs):
        y = self.convolution_1(inputs)
        y = self.batch_norm_1(y)
        y = self.activation_1(y)
        y = self.convolution_2(y)
        y = self.batch_norm_2(y)
        return self.addition([inputs, y])


class DiscriminatorConvolution(tf.keras.layers.Layer):

    def __init__(self, filters, strides=1, use_batch_norm=True):
        super(DiscriminatorConvolution, self).__init__()
        self.use_batch_norm = use_batch_norm
        self.convolution_1 = tf.keras.layers.Conv2D(
            filters, kernel_size=3,
            strides=strides, padding='same'
        )
        if self.use_batch_norm:
            self.batch_norm = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.activation = tf.keras.layers.LeakyReLU(alpha=0.2)

    def call(self, inputs):
        y = self.convolution_1(inputs)
        y = self.batch_norm(y) if self.use_batch_norm else y
        y = self.activation(y)
        return y
