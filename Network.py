from keras.layers.convolutional import UpSampling2D
from keras.layers.preprocessing.image_preprocessing import HORIZONTAL
from keras.models import Model
from keras.layers import (
    Input,
    Dense,
    Conv2D,
    Conv2DTranspose,
    MaxPool2D,
    AveragePooling2D,
    ReLU,
    PReLU,
    LeakyReLU,
    BatchNormalization,
    Concatenate,
    Add,
    Reshape,
    Flatten,
    Dropout,
    RandomFlip,
)
from keras.activations import tanh
from keras.initializers import RandomNormal
from tensorflow.keras.optimizers import Adam
from discrimination import MinibatchDiscrimination

import tensorflow as tf
import keras as K
import os

# If uncommented, forces the use of the cpu or gpu
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Hides tensorflow outputs and warnings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # or any {'0', '1', '2'}
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Prevents memory overflow errors
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)


def init():
    return RandomNormal(mean=0.0, stddev=0.02)


def optimizer():
    return Adam(lr=0.0002, beta_1=0.5)  # standard lr: 0.0002


def create_generator(features_g):
    ### Parameters ###
    features_g = features_g

    ### Create Generator ###
    # Input Layer
    inputX = Input(shape=(128))

    # Reshape
    x = Dense(4 * 4 * features_g * 16)(inputX)
    x = PReLU()(x)
    x = Reshape((4, 4, features_g * 16))(x)

    x = generator_block(x, filters=features_g * 8)
    x = generator_block(x, filters=features_g * 4)
    x = generator_block(x, filters=features_g * 2)
    x = generator_block(x, filters=features_g * 1)

    # for _ in range(5):
    #    residual_block_add(x, filters=features_g * 1, kernel_size=3)

    # Output Layer
    x = Conv2D(filters=3, kernel_size=1, padding="same", activation="tanh")(x)
    model = Model(inputs=inputX, outputs=x, name="Generator")

    return model


def residual_block_add(inputX, filters, kernel_size):
    """
    x
		|\
		| \
		|  conv2d
		|  activation
		|  conv2d
        |  (multiply scaling)
		| /
		|/
		+ (addition here)
		|
		result
    """
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(inputX)
    x = PReLU()(x)
    x = Conv2D(filters=filters, kernel_size=kernel_size, padding="same")(x)
    x *= 0.1
    x = Add()([inputX, x])

    return x


def generator_block(inputX, filters):
    x = UpSampling2D()(inputX)
    x = Conv2D(filters=filters, kernel_size=3, padding="same")(x)
    x = PReLU()(x)

    return x


def create_discriminator(features_d):
    ### Parameters ###
    features_d = features_d

    ### Create Discriminator ###
    # Input Layer
    inputX = Input(shape=(64, 64, 3))

    # x = Conv2D(filters=features_d * 1, kernel_size=1, padding="same")(inputX)

    # for _ in range(5):
    #    residual_block_add(x, filters=features_d * 1, kernel_size=3)

    # Downscale
    x = discriminator_block(inputX, filters=features_d * 1)
    x = discriminator_block(x, filters=features_d * 2)
    x = discriminator_block(x, filters=features_d * 4)
    x = discriminator_block(x, filters=features_d * 8)
    x = discriminator_block(x, filters=features_d * 16)

    # Output Layers
    x = Flatten()(x)
    x = MinibatchDiscrimination(nb_kernels=5, kernel_dim=3)(x)
    x = PReLU()(x)
    # x = Dense(1024, kernel_initializer=init())(x)
    # x = PReLU()(x)
    x = Dense(1, activation="sigmoid")(x)

    ### Compile Discriminator ###
    model = Model(inputs=inputX, outputs=x, name="Discriminator")
    model.compile(optimizer=optimizer(), loss="binary_crossentropy")

    return model


def discriminator_block(inputX, filters):
    x = Conv2D(filters=filters, kernel_size=4, strides=2, padding="same")(inputX)
    x = PReLU()(x)

    return x


def create_gan(discriminator, generator):
    discriminator.trainable = False

    gan_input = Input(shape=(128))
    x = generator(gan_input)

    gan_output = discriminator(x)

    gan = Model(
        inputs=gan_input,
        outputs=gan_output,
        name="Generative_Adversarial_Network",
    )

    gan.compile(optimizer=optimizer(), loss="binary_crossentropy")

    return gan


if __name__ == "__main__":
    g = create_generator(16)
    g.summary()

    d = create_discriminator(16)
    d.summary()

    gan = create_gan(d, g)
    gan.summary()
