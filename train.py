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
        self.pre_train_checkpoint, self.pre_train_checkpoint_manager = get_checkpoint_pretrain(
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
            self.pre_train_checkpoint_manager.save()
            print(
                'Pre-trained Generator saved at {}'.format(
                    self.pre_train_checkpoint_manager.latest_checkpoint
                )
            )

    @tf.function
    def train_step(self, x, y):
        valid = tf.ones((x.shape[0],) + self.disc_patch)
        fake = tf.zeros((x.shape[0],) + self.disc_patch)
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            fake_hr = self.generator(x)
            valid_prediction = self.discriminator(y)
            fake_prediction = self.discriminator(fake_hr)
            c_loss = content_loss(y, fake_hr)
            adv_loss = 1e-3 * tf.keras.losses.BinaryCrossentropy()(valid, fake_prediction)
            mse_loss = tf.keras.losses.MeanSquaredError()(y, fake_hr)
            perceptual_loss = c_loss + adv_loss + mse_loss
            valid_loss = tf.keras.losses.BinaryCrossentropy()(valid, valid_prediction)
            fake_loss = tf.keras.losses.BinaryCrossentropy()(fake, fake_prediction)
            d_loss = tf.add(valid_loss, fake_loss)
        gen_grads = gen_tape.gradient(perceptual_loss, self.generator.trainable_variables)
        self.g_optimizer.apply_gradients(zip(gen_grads, self.generator.trainable_variables))
        disc_grads = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)
        self.d_optimizer.apply_gradients(zip(disc_grads, self.discriminator.trainable_variables))
        return d_loss, adv_loss, c_loss, mse_loss

    def train(self):
        self.checkpoint, self.checkpoint_manager = get_checkpoint_train(
            [self.generator, self.discriminator],
            [self.g_optimizer, self.d_optimizer],
            checkpoint_dir=self.config['train_checkpoint_dir']
        )
        self.train_summary_writer = tf.summary.create_file_writer(self.config['train_logdir'])
        with self.train_summary_writer.as_default():
            iterations = 1
            for epoch in range(1, self.config['train_epochs'] + 1):
                print('Epoch: {}'.format(epoch))
                for x, y in tqdm(self.dataset):
                    disc_loss, adv_loss, c_loss, mse_loss = self.train_step(x, y)
                    denorm_x = tf.cast(255 * x, tf.uint8)
                    denorm_y = tf.cast(255 * (y + 1.0) / 2.0, tf.uint8)
                    denorm_pred = tf.cast(255 * (self.generator.predict(x) + 1.0) / 2.0, tf.uint8)
                    psnr_batch = []
                    for i in range(self.config['batch_size']):
                        psnr_batch.append(get_psnr(denorm_y[i], denorm_pred[i]))
                    psnr = sum(psnr_batch) / self.config['batch_size']
                    tf.summary.scalar('train/adversarial_loss', adv_loss, step=iterations)
                    tf.summary.scalar('train/content_loss', content_loss, step=iterations)
                    tf.summary.scalar('train/mse_loss', mse_loss, step=iterations)
                    tf.summary.scalar('train/discriminator_loss', disc_loss, step=iterations)
                    tf.summary.scalar(
                        'learning_rate/learning_rate_G',
                        self.g_optimizer.lr(iterations),
                        step=iterations
                    )
                    tf.summary.scalar(
                        'learning_rate/learning_rate_D',
                        self.d_optimizer.lr(iterations),
                        step=iterations
                    )
                    tf.summary.scalar('train/psnr', psnr, step=iterations)
                    tf.summary.image('Low Res', denorm_x, step=iterations)
                    tf.summary.image('High Res', denorm_y, step=iterations)
                    tf.summary.image('Generated', denorm_pred, step=iterations)
                    self.train_summary_writer.flush()
                    iterations += 1
                if epoch % self.config['save_interval'] == 0:
                    self.checkpoint_manager.save()
                    print(
                        'Model Checkpoints saved at {}'.format(
                            self.checkpoint_manager.latest_checkpoint
                        )
                    )