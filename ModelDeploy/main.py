import argparse
from argparse import Namespace
from tqdm import tqdm
import cv2
import os
import structures as st
import json
import numpy as np
import models as M
import torch
import albumentations as A
import albumentations.pytorch.transforms as tf
import utils as ut
import math

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description='main deploy script')
    parser.add_argument('--path', help='images|video', type=str, default='../mmdetection3d/data/kitti/testing/image_2/')
    parser.add_argument('--file_format', help='images|video', type=str, default='%06d.png')
    args = parser.parse_args()
    return args

def load_json(path:str) -> dict:
    with open(path, 'r') as file:
        config = json.load(file)
    return config

def create_transform():
    return A.Compose([
        A.Resize(384,1280),
        A.Normalize(),
        tf.ToTensorV2(),
    ])
    

def main(args:Namespace):
    cap = cv2.VideoCapture(os.path.join(args.path, args.file_format))
    images = [f for f in os.listdir(args.path) if f.endswith('.png')]
    config = load_json('./config.json')
    render_manager = st.RenderManager()
    kitti_coordinate_converter = st.CoordinateConverter(cam2img=np.array(config['kitti']['matrix_camera_to_image']))
    transform = create_transform()
    model = M.MMSmoke('../mmdetection3d/checkpoints/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth')
    
    # inference loop
    pbar = tqdm(range(len(images)))
    progress = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        input_data = transform(image=frame)['image']
        input_data = input_data.to('cuda')

        # inference
        inference_result = model.forward(input_data)
        boxes = ut.create_bbox3d(inference_result)
        pboxes = [ut.project_bbox3d(kitti_coordinate_converter, box) for box in boxes]
        for pbbox in pboxes:
            render_manager.draw_projected_box3d(frame, pbbox.raw_points)
        cv2.imwrite('work_dirs/frame.png', frame)

        # update
        pbar.set_description(f'grab : {ret}')

        # update progress
        pbar.update(1)
        progress += 1
        if(progress == len(images)): break
    return 0

if __name__ == "__main__":
    args = parse_args()
    main(args)
    