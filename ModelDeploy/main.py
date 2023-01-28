import argparse
from argparse import Namespace
from tqdm import tqdm
import cv2
import os


def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description='main deploy script')
    parser.add_argument('--path', help='images|video', type=str, default='../mmdetection3d/data/kitti/testing/image_2/')
    parser.add_argument('--file_format', help='images|video', type=str, default='%06d.png')
    args = parser.parse_args()
    return args

def main(args:Namespace):
    cap = cv2.VideoCapture(os.path.join(args.path, args.file_format))
    images = [f for f in os.listdir(args.path) if f.endswith('.png')]
    pbar = tqdm(range(len(images)))
    progress = 0
    while(cap.isOpened()):
        ret, frame = cap.read()

        # update
        pbar.set_description(f'result : {ret}')

        # update progress
        pbar.update(1)
        progress += 1
        if(progress == len(images)): break
    return 0

if __name__ == "__main__":
    args = parse_args()
    main(args)
    