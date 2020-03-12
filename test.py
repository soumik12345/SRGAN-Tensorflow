from src.models import *

# x = tf.random.normal((1, 64, 64, 3))
# model = Generator()
# y = model(x)
# print(y.shape)

x = tf.random.normal((1, 256, 256, 3))
model = Discriminator()
y = model(x)
print(y.shape)
