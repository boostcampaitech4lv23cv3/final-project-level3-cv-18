import cv2
import os
import json
import math
import torch
import argparse
import numpy as np
from tqdm import tqdm
from argparse import Namespace
import albumentations as A
import albumentations.pytorch.transforms as tf
import models as M
import utils as ut
import modules as md

def parse_args() -> Namespace:
    parser = argparse.ArgumentParser(
        description='main deploy script')
    parser.add_argument('--path', help='images|video', type=str, default='./mmdetection3d/data/kitti/testing/image_2/')
    parser.add_argument('--file_format', help='images|video', type=str, default='%06d.png')
    # parser.add_argument('--path', help='images|video', type=str, default='./work_dirs/1456_resize')
    # parser.add_argument('--file_format', help='images|video', type=str, default='%d.png')
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
    loader = md.DataLoaderCV(os.path.join(args.path, args.file_format))
    config = load_json('./ModelDeploy/config.json')
    render_manager = md.RenderManager()
    kitti_coordinate_converter = md.CoordinateConverter(cam2img=np.array(config['kitti']['matrix_camera_to_image']))
    transform = create_transform()
    model = M.MMSmoke('./mmdetection3d/checkpoints/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth')
    
    # inference loop
    i=0
    pbar = tqdm(range(loader.frame_count))
    while(loader.is_progress):
        ret, frame = loader.get_frame()
        input_data = transform(image=frame)['image']
        input_data = input_data.to('cuda')

        # inference
        inference_result = model.forward(input_data)
        levels = ut.check_danger(inference_result)
        boxs = ut.create_bbox3d(inference_result)
        pbboxs = ut.project_bbox3ds(kitti_coordinate_converter, boxs)
        ut.render_pbboxs(frame, render_manager, pbboxs, levels)
        cv2.imwrite('work_dirs/frame.png', frame)

        # update
        pbar.set_description(f'grab : {ret}')

        # update progress
        pbar.update(1)

        if i > 0:
            if input() == "a":
                break
            else:
                i+=1
        else:
            i+=1
    return 0

if __name__ == "__main__":
    args = parse_args()
    main(args)
    