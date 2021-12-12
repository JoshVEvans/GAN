import os
import cv2
from tqdm import tqdm

# Video Generating function
def generate_video():
    image_folder = "GAN_plots/"  # make sure to use your folder
    video_name = "video.mov"
    # os.chdir("C:\\Python\\Geekfolder2")

    images = [
        img
        for img in os.listdir(image_folder)
        if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith("png")
    ]

    frames = []
    for image in tqdm(images):
        frames.append(cv2.imread(f"{image_folder}{image}"))

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frames[0].shape

    video = cv2.VideoWriter(video_name, fourcc, 12, (width, height))

    for frame in tqdm(frames):
        video.write(frame)


if __name__ == "__main__":
    generate_video()
