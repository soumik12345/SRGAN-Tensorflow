import json
import os
import tensorflow as tf
from PIL import Image


def get_psnr(img1, img2):
    return tf.image.psnr(img1, img2, max_val=255).numpy()


def get_checkpoint_pretrain(model, optimizer, checkpoint_dir='./pretrain_checkpoints/'):
    checkpoint = tf.train.Checkpoint(
        step=tf.Variable(0, name='step'),
        optimizer=optimizer, model=model
    )
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint=checkpoint,
        directory=checkpoint_dir,
        max_to_keep=3
    )
    if checkpoint_manager.latest_checkpoint:
        checkpoint.restore(checkpoint_manager.latest_checkpoint)
        print(
            'Loaded ckpt from {} at step {}.'.format(
                checkpoint_manager.latest_checkpoint,
                checkpoint.step.numpy()
            )
        )
    else:
        print("Training from scratch....")
    return checkpoint, checkpoint_manager


def set_memory_growth():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print("Detect {} Physical GPUs, {} Logical GPUs.".format(len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)


def parse_config(json_file):
    with open(json_file, 'r') as f:
        configs = json.load(f)
    return configs


def cache_image(image, location, file_id, crop_size=256, stride=64):
    height, width = image.shape[:2]
    i, j, num = 0, 0, 0
    while i <= width - crop_size:
        while j <= height - crop_size:
            num += 1
            crop = Image.fromarray(image[i:i + 256, j:j + 256, :])
            crop.save(os.path.join(location, str(file_id) + str(num)))
            j += stride
        i += stride
