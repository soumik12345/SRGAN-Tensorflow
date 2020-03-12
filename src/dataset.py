import os, math
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE


class SRDataset:

    def __init__(self, images, image_size, downsample_scale):
        self.images = images
        self.image_size = image_size
        self.downsample_scale = downsample_scale

    def load_image(self, image_file):
        image = tf.io.read_file(image_file)
        image = tf.io.decode_png(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = self.random_crop(image)
        return image

    def random_crop(self, image):
        return tf.image.random_crop(image, [self.image_size, self.image_size, 3])

    def get_pair(self, image):
        return tf.image.resize(
            image,
            [self.image_size // self.downsample_scale, ] * 2,
            method='bicubic'
        ), image

    def normalize(self, lr_image, hr_image):
        hr_image = hr_image * 2.0 - 1.0
        return lr_image, hr_image

    @staticmethod
    def denormalize(image):
        return (image + 1.0) / 2.0

    def get_dataset(self, batch_size, buffer_size):
        dataset = tf.data.Dataset.from_tensor_slices(self.images)
        dataset = dataset.map(
            self.load_image,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(
            self.random_crop,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(
            self.get_pair,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.map(
            self.normalize,
            num_parallel_calls=AUTOTUNE
        )
        dataset = dataset.shuffle(buffer_size).batch(
            batch_size, drop_remainder=True
        ).prefetch(AUTOTUNE)
        return dataset

    @staticmethod
    def visualize_batch(dataset, model=None):
        x_batch, y_batch = next(iter(dataset))
        x_batch = x_batch.numpy()
        y_batch = y_batch.numpy()
        c = 0
        if model is None:
            fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 16))
            plt.setp(axes.flat, xticks=[], yticks=[])
            for i, ax in enumerate(axes.flat):
                if i % 2 == 0:
                    ax.imshow(x_batch[c])
                    ax.set_xlabel('Low_Res_' + str(c + 1))
                elif i % 2 == 1:
                    ax.imshow(SRDataset.denormalize(y_batch[c]))
                    ax.set_xlabel('High_Res_' + str(c + 1))
                    c += 1
        else:
            fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 16))
            plt.setp(axes.flat, xticks=[], yticks=[])
            for i, ax in enumerate(axes.flat):
                if i % 3 == 0:
                    ax.imshow(x_batch[c])
                    ax.set_xlabel('Low_Res_' + str(c + 1))
                elif i % 3 == 1:
                    ax.imshow(SRDataset.denormalize(y_batch[c]))
                    ax.set_xlabel('High_Res_' + str(c + 1))
                elif i % 3 == 2:
                    ax.imshow(np.squeeze(SRDataset.denormalize(model(np.expand_dims(x_batch[c], axis=0)))))
                    ax.set_xlabel('High_Res_' + str(c + 1))
                    c += 1
        plt.show()
