import Utils
import Network

from keras import backend as K

import multiprocessing as mp
from tqdm import trange
import numpy as np
import random
import cv2

import os

dim = 64


def get_image(image_path):

    # Read image from path
    image = cv2.imread(image_path)
    # Resize Image
    # if image.shape[0] > dim and image.shape[1] > dim:
    image = center_crop(image, (171, 171))
    image = cv2.resize(image, (dim, dim), interpolation=cv2.INTER_CUBIC)
    # else:
    # image = cv2.resize(image, (dim, dim), interpolation=cv2.INTER_NEAREST)
    # Horizontal Flipping
    if random.choice([True, False]):
        image = cv2.flip(image, 1)

    return image


def center_crop(img, dim):
    width, height = img.shape[1], img.shape[0]
    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]
    mid_x, mid_y = width // 2, height // 2
    cw2, ch2 = crop_width // 2, crop_height // 2
    crop_img = img[mid_y - ch2 : mid_y + ch2, mid_x - cw2 : mid_x + cw2]

    return crop_img


def training(dataset_paths, epochs=1, batch_size=128, steps_per_epoch=1000):
    # Load multiple paths
    image_path_temp = []
    for dataset_path in dataset_paths:
        image_paths = list(os.listdir(dataset_path))
        image_paths = [f"{dataset_path}{image_path}" for image_path in image_paths]

        image_path_temp = [y for x in [image_path_temp, image_paths] for y in x]

    image_paths = np.array(image_path_temp)

    # Create GAN
    generator = Network.create_generator(16)
    generator.summary()
    discriminator = Network.create_discriminator(16)
    discriminator.summary()
    gan = Network.create_gan(discriminator, generator)
    gan.summary()

    # Parameters
    TTUR = True  # Two Time-Scale Update Rule
    latent_space = 128
    discriminator_train_steps = 1  # Number of times to train discriminator
    generator_train_steps = 1  # Number of times to train discriminator

    # Metrics
    metrics = []

    # Create test noise
    num_examples = 100
    test_noise = np.random.normal(loc=0, scale=1, size=[num_examples, latent_space])

    # discriminator b and generator a
    if TTUR:
        K.set_value(discriminator.optimizer.learning_rate, 1e-4)
        K.set_value(gan.optimizer.learning_rate, 2e-4)

    # Multiprocessing
    workers = mp.cpu_count()
    if batch_size <= mp.cpu_count():
        workers = batch_size
    print(f"Workers: {workers}")
    p = mp.Pool(workers)

    for e in range(1, epochs):
        print("-" * 25 + f"Epoch: {e}/{epochs}" + "-" * 25)

        # Parameters
        generator_loss = 0
        discriminator_loss = 0

        # Constants
        # Tricking the noised input of the Generator as real data
        y_fake = np.zeros(shape=batch_size)
        y_gen = np.ones(batch_size)

        tr = trange(steps_per_epoch, desc=f"Epoch: {e}", leave=True)
        for i in tr:
            y_real = np.full(shape=batch_size, fill_value=0.9 + random.random() / 20)

            ### Train Discriminator ###
            discriminator.trainable = True
            for _ in range(discriminator_train_steps):
                ### Discriminator Input ###
                # Get Real Images
                batch_paths = image_paths[
                    np.random.randint(0, len(image_paths), size=(batch_size))
                ]
                X_real = (np.array(list(p.map(get_image, batch_paths))) - 127.5) / 127.5

                # Get Fake Images
                noise = np.random.normal(0, 1, [batch_size, latent_space])
                # Generate Fake Images
                X_fake = generator(noise)

                discriminator_loss += discriminator.train_on_batch(X_real, y_real)
                discriminator_loss += discriminator.train_on_batch(X_fake, y_fake)
            discriminator.trainable = False  # Ensures that discriminator remains static during training of generator in GAN

            ### Train Generator ###
            # Multiple GAN passes
            for _ in range(generator_train_steps):
                noise = np.random.normal(0, 1, [batch_size, latent_space])
                generator_loss += gan.train_on_batch(noise, y_gen)

            # Update Progress Bar
            tr.set_postfix(
                d_loss=discriminator_loss / 2 / discriminator_train_steps / (i + 1),
                g_loss=generator_loss / generator_train_steps / (i + 1),
            )

        # Analysis
        discriminator_loss = (
            discriminator_loss / 2 / discriminator_train_steps / steps_per_epoch
        )
        generator_loss = generator_loss / generator_train_steps / steps_per_epoch

        Utils.plot_generated_images(
            e,
            discriminator_loss,
            generator,
            noise=test_noise,
            dim=(10, 10),
            figsize=(10, 10),
            image_dim=dim,
        )

        metrics.append([e, discriminator_loss, generator_loss])
        np.save("GAN/GAN_plots/metrics", np.array(metrics))
        # generator.save(f"weights/GAN/{e}-{generator_loss}.hdf5")
        generator.save(f"weights/GAN/{e}.hdf5")


if __name__ == "__main__":
    # Parameters
    dataset_paths = ["data/train/CELEBA/"]
    epochs = 10000
    batch_size = 64
    steps_per_epoch = 250

    training(
        dataset_paths=dataset_paths,
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
    )
