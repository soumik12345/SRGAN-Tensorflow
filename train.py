import logging
from tqdm import tqdm
from glob import glob
import tensorflow as tf
from src.loss import *
from src.utils import *
from src.models import *
from src.dataset import *


class Trainer:

    def __init__(self, config_file):
        self.config = parse_config(config_file)
        logging.info('Setting Memory Growth')
        set_memory_growth()
        logging.info('Creating Dataset Object')
        self.dataset = self.get_dataset()
        self.generator, self.discriminator, self.vgg = self.get_models()
        self.g_schedule, self.d_schedule = self.get_lr_schedulers()
        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=self.g_schedule)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=self.d_schedule)
        patch = int(self.config['hr_patch_size'] / 2 ** self.config['downsample_scale'])
        disc_patch = (patch, patch, 1)

    def get_dataset(self):
        super_resolution_dataset = SRDataset(
            glob(self.config['hr_img_path']),
            self.config['hr_patch_size'],
            self.config['downsample_scale']
        )
        dataset = super_resolution_dataset.get_dataset(
            self.config['batch_size'],
            self.config['buffer_size']
        )
        return dataset

    def get_models(self):
        generator = Generator(
            filters=self.config['models']['generator']['filters'],
            n_res_blocks=self.config['models']['generator']['n_res_blocks']
        )
        discriminator = Discriminator(self.config['models']['discriminator']['filters'])
        vgg = get_vgg_model(
            (
                self.config['hr_patch_size'],
                self.config['hr_patch_size'],
                self.config['n_channels']
            )
        )
        vgg.trainable = False
        return generator, discriminator, vgg

    def get_lr_schedulers(self):
        g_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.config['lr_schedulers']['generator']['initial_lr'], staircase=True,
            decay_steps=self.config['lr_schedulers']['generator']['decay_steps'],
            decay_rate=self.config['lr_schedulers']['generator']['decay_rate'],
        )
        d_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            self.config['lr_schedulers']['discriminator']['initial_lr'], staircase=True,
            decay_steps=self.config['lr_schedulers']['discriminator']['decay_steps'],
            decay_rate=self.config['lr_schedulers']['discriminator']['decay_rate'],
        )
        return g_schedule, d_schedule

    @tf.function
    def generator_pretrain_step(self, x, y):
        with tf.GradientTape() as tape:
            fake_hr = self.generator(x)
            loss_mse = tf.keras.losses.MeanSquaredError()(y, fake_hr)
        grads = tape.gradient(loss_mse, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_variables))
        return loss_mse

    def pretrain(self):
        self.checkpoint, self.checkpoint_manager = get_checkpoint_pretrain(
            self.generator, self.g_optimizer,
            checkpoint_dir=self.config['pretrain_checkpoint_dir']
        )
        self.pretrain_summary_writer = tf.summary.create_file_writer(self.config['pretrain_logdir'])
        with self.pretrain_summary_writer.as_default():
            iteration = 0
            for epoch in range(1, self.config['pretrain_epochs'] + 1):
                print('Epoch: {}'.format(epoch))
                for x, y in tqdm(self.dataset):
                    loss = self.generator_pretrain_step(x, y)
                    tf.summary.scalar('pretrain/mse_loss', loss, step=tf.cast(iteration, tf.int64))
                    self.pretrain_summary_writer.flush()
                    iteration += 1
            self.checkpoint_manager.save()
            print(
                'Pre-trained Generator saved at {}'.format(
                    self.checkpoint_manager.latest_checkpoint
                )
            )
