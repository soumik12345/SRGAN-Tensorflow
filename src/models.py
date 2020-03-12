from .blocks import *
import tensorflow as tf


class Generator(tf.keras.Model):

    def __init__(self, filters=64, n_res_blocks=16):
        super(Generator, self).__init__()
        self.n_res_blocks = n_res_blocks
        self.convolution_1 = tf.keras.layers.Conv2D(filters, 9, padding='same')
        self.activation = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.residual_blocks = [Residual(filters)] * self.n_res_blocks
        self.convolution_2 = tf.keras.layers.Conv2D(filters, 3, padding='same')
        self.batch_norm = tf.keras.layers.BatchNormalization()
        self.addition = tf.keras.layers.Add()
        self.upsample_1 = UpSample(filters * 4)
        self.upsample_2 = UpSample(filters * 4)
        self.convolution_out = tf.keras.layers.Conv2D(3, 9, padding='same', activation='tanh')

    def call(self, inputs):
        y = self.convolution_1(inputs)
        y = self.activation(y)
        y_sec = self.activation(y)
        for i in range(self.n_res_blocks):
            y = self.residual_blocks[i](y)
        y = self.convolution_2(y)
        y = self.batch_norm(y)
        y = self.addition([y_sec, y])
        y = self.upsample_1(y)
        y = self.upsample_2(y)
        y = self.convolution_out(y)
        return y


class Discriminator(tf.keras.Model):

    def __init__(self, filters=32):
        super(Discriminator, self).__init__()
        self.blocks = [
            DiscriminatorConvolution(filters, use_batch_norm=False),
            DiscriminatorConvolution(filters, strides=2),
            DiscriminatorConvolution(filters),
            DiscriminatorConvolution(filters, strides=2),
            DiscriminatorConvolution(filters * 2),
            DiscriminatorConvolution(filters * 2, strides=2),
            DiscriminatorConvolution(filters * 2),
            DiscriminatorConvolution(filters * 2, strides=2),
            tf.keras.layers.Conv2D(
                1, kernel_size=1, strides=1,
                activation='sigmoid', padding='same'
            )
        ]

    def call(self, inputs):
        y = inputs
        for block in self.blocks:
            y = block(y)
        return y
