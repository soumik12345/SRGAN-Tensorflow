import tensorflow as tf


def get_vgg_model(input_shape):
    vgg_19 = tf.keras.applications.VGG19(
        weights="imagenet",
        input_shape=input_shape,
        include_top=False
    )
    vgg_19.trainable = False
    for layer in vgg_19.layers:
        layer.trainable = False
    return tf.keras.models.Model(
        inputs=vgg_19.input,
        outputs=vgg_19.get_layer("block5_conv4").output
    )


@tf.function
def content_loss(hr, sr, vgg):
    sr = tf.keras.applications.vgg19.preprocess_input(((sr + 1.0) * 255) / 2.0)
    hr = tf.keras.applications.vgg19.preprocess_input(((hr + 1.0) * 255) / 2.0)
    sr_features = vgg(sr) / 12.75
    hr_features = vgg(hr) / 12.75
    return tf.keras.losses.MeanSquaredError()(hr_features, sr_features)
