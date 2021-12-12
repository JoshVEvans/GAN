import os
import cv2
from tqdm import tqdm

# Video Generating function
def generate_video():
    image_folder = "GAN_plots/"  # make sure to use your folder
    video_name = "md_files/video.mp4"
    # os.chdir("C:\\Python\\Geekfolder2")

    images = [
        img
        for img in os.listdir(image_folder)
        if img.endswith(".jpg") or img.endswith(".jpeg") or img.endswith("png")
    ]

    frames = []
    for i, image in tqdm(enumerate(images)):
        if i % 2 == 0:
            frame = cv2.imread(f"{image_folder}{image}")
            frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_LANCZOS4)
            frames.append(frame)

    fourcc = cv2.VideoWriter_fourcc("m", "p", "4", "v")

    # setting the frame width, height width
    # the width, height of first image
    height, width, layers = frames[0].shape

    video = cv2.VideoWriter(video_name, fourcc, 12, (width, height))

    for frame in tqdm(frames):
        video.write(frame)


if __name__ == "__main__":
    generate_video()
