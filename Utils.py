from keras.datasets import mnist


import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import cv2
import os

dim = 32
dataset_name = "Cryptopunks"


def load_data():
    try:
        x_train = np.load(f"data\stored\{dataset_name}-{dim}x{dim}.npy")
        return x_train
    except:
        data_path = f"data/train/{dataset_name}/"
        image_names = list(os.listdir(data_path))
        # print(len(image_names))
        x_train = []
        for image_name in tqdm(image_names):
            image = np.array(cv2.imread(f"{data_path}{image_name}")).astype(np.float32)
            if len(image.shape) == 3:
                image = cv2.resize(image, (32, 32), interpolation=cv2.INTER_LANCZOS4)
                x_train.append(image)

                image = cv2.flip(image, 1)
                x_train.append(image)

        # Format and Normalize Data
        x_train = np.array(x_train)
        x_train = (x_train - 127.5) / 127.5

        # Save data
        np.save(f"data/stored/{dataset_name}-64x64.npy", x_train)
        return x_train


def augment_images(images):
    X = []

    # Create X data
    for image in images:
        # Horizontal Flipping
        if random.choice([True, False]):
            image = cv2.flip(image, 1)

        X.append(image)

    # Format Data
    X = np.array(X)

    return X


def load_batch(image_paths, dim):
    images = []
    for image_path in image_paths:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (dim, dim), interpolation=cv2.INTER_LANCZOS4)

    images = (images - 127.5) / 127.5


def get_image(image_path):

    # Read image from path
    image = cv2.imread(image_path)
    # Resize Image
    image = cv2.resize(image, (128, 128), interpolation=cv2.INTER_LANCZOS4)

    # Random Horizontal Flipping
    if random.choice([True, False]):
        image = cv2.flip(image, 1)

    return image


def unison_shuffle(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


def plot_generated_images(
    epoch, discriminator_loss, generator, noise, dim, figsize, image_dim
):
    generated_images = generator.predict(noise)
    generated_images = (generated_images * 127.5 + 127.5) / 255
    generated_images = generated_images.reshape(noise.shape[0], image_dim, image_dim, 3)

    for i in range(generated_images.shape[0]):
        generated_images[i] = cv2.cvtColor(generated_images[i], cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=figsize)

    for i in range(noise.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation="nearest")
        plt.axis("off")
    plt.tight_layout()
    # plt.savefig(f"GAN/GAN_plots/GAN_Epoch-{epoch}-{discriminator_loss}.png")
    plt.savefig(f"GAN/GAN_plots/GAN_Epoch-{epoch}.png")
    plt.savefig(f"GAN/GAN_plots/GAN_Epoch-0.png")

    plt.close(fig)
