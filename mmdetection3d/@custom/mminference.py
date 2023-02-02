# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
from time import sleep
from math import cos, sin, radians, degrees

from mmengine.config import Config, DictAction
from mmengine.registry import RUNNERS
from mmengine.runner import Runner
from mmdet3d.utils import register_all_modules, replace_ceph_backend
from mmdet3d.models.detectors import SMOKEMono3D

import torch
import numpy as np
import cv2
import sys
sys.path.append('..')
import json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='MMDet3D test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args

def points_cam2img(points_3d:np.ndarray, proj_mat:np.ndarray) -> np.ndarray:
    """Project points in camera coordinates to image coordinates.

    Args:
        points_3d (np.ndarray): Points in shape (N, 3)
        proj_mat (np.ndarray):
            Transformation matrix between coordinates.

    Returns:
        np.ndarray: Points in image coordinates,
            with shape [N, 2].
    """
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1 
    
    points_4 = np.hstack([points_3d, np.ones(points_shape, points_3d.dtype)])#torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)
    point_2d = points_4 @ proj_mat.T
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    return point_2d_res

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

# 차량 주의 색 표시
# def warning_color(object_points):
#     # 거리별 오브젝트 위험도를 색으로 나타내준다.
#     code_red = 30 # Critical!
#     code_orange = 60 # Warning!
#     color_list = []
#     d_pred_list = []

#     for point in object_points:
#         point = point/10
#         d = (point[0]**2+point[1]**2)**(1/2)
#         if d <= code_red:
#             color_list.append([0,0,255])
#         elif d <= code_orange:
#             color_list.append([153,0,255])
#         else:
#             color_list.append([0,0,0])

#     return color_list, d_pred_list

def rotate_new(x0, y0, theta):
    # 중점과 좌표를 주면 회전값 반환
    x1 = (x0 - xm) * cos(radians(360-degrees(theta))) - (y0 - ym) * sin(radians(360 - delattr(theta))) + xm
    y1 = (x0 - xm) * sin(radians(360-degrees(theta))) - (y0 - ym) * cos(radians(360 - delattr(theta))) + xm

    return int(x1), int(y1)

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    # cv2.drawContours(image, [[qs[0].tolist(),qs[3].tolist(),qs[7].tolist(),qs[4].tolist()]], -1, (255,0,0))
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def render_result(image:np.ndarray, cam2img:list, bboxes:np.ndarray, labels:np.ndarray, scores:np.ndarray) -> np.ndarray:
    # cv Image 변수 새로 선언 
    points=[]
    points_wl=[]
    yaw_list = []
    for idx, (bbox, label, score) in enumerate(zip(bboxes.tolist(), labels.tolist(), scores.tolist())):
        # Each row is (x, y, z, x_size, y_size, z_size, yaw)
        yaw = bbox[6]
        yaw_list.append(yaw)
        rotation_metrix = roty(bbox[6])
        print(f'{idx}_rotation_metrix:' , rotation_metrix)
        w = bbox[3]
        h = bbox[4]
        l = bbox[5]
        x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        y_corners = [-h, -h, -h, -h, 0, 0, 0, 0]
        z_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        corners_3d = np.dot(rotation_metrix, np.vstack([x_corners, y_corners, z_corners])).astype(np.double)
        print('corner----------------------')
        print(corners_3d)
        corners_3d[0, :] = corners_3d[0, :] + bbox[0] # type: ignore
        corners_3d[1, :] = corners_3d[1, :] + bbox[1] # type: ignore
        corners_3d[2, :] = corners_3d[2, :] + bbox[2]  # type: ignore
        uv_origin = points_cam2img(np.transpose(corners_3d), np.array(cam2img))
        corners_2d = (uv_origin - 1).round()
        draw_projected_box3d(image, 
                             corners_2d, 
                             color=(255 - int(200 * (label/3.)), 200+int(55 * score), int(200 * (label/3.))),
                             thickness=1+int(3 * score)
                            ) # type: ignore
        #좌표 변환 포인트 찍기(corners_3d[0, :], corners_3d[2, :])        
        points.append([corners_3d[0, :][:-4].astype(np.float32).tolist(),corners_3d[2, :][:-4].astype(np.float32).tolist()])
        points_wl.append([l,w])
        #points.append([int(bbox[0]),int(bbox[2])])
        
    #print(f'{idx}-points : {points}')
    #print('corner3d[0]:',corners_3d[0, :])
    print('points--------------------')
    print(points)
                      
    return image, points

def render_map(image, points, point_color = (0,0,0)):

    (x,y)=(image.shape[1]//2 ,image.shape[0])
    #draw circle
    color_spec=[[0,0,255],[153,0,255],[0,153,255],[0,204,255],[153,255,0],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51]]
    k=0
    for r in range(50, 700, 100):
        cv2.circle(image, (x,y), r, color_spec[k], thickness=3)
        k+=1

    for p in points:
            # p : [corners_3d[0, :][:-4].astype(np.int).tolist(),corners_3d[2, :][:-4].astype(np.int).tolist()]
            # P : 2x4 array
            
            rectpoints = np.array(p).T
            rectpoints = rectpoints * 10
            rectpoints[:,0] = rectpoints[:,0] + 250
            rectpoints[:,1] = 500 -1 * rectpoints[:,1]
            rectpoints_list = rectpoints.astype(np.int32)
            cv2.polylines(image, [rectpoints_list], True, (0,0,0), thickness=4,lineType=cv2.LINE_AA)
            print('rec_list:',rectpoints_list)
            cv2.fillConvexPoly(image, rectpoints_list, (0,0,0))
            
    print(type(rectpoints_list))
    return image

def main():
    args = parse_args()

    register_all_modules(init_default_scope=False)
    cfg = Config.fromfile(args.config)

    cfg.launcher = 'none'
    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    cfg.load_from = args.checkpoint
    runner = Runner.from_cfg(cfg)
    runner.load_or_resume()

    model:SMOKEMono3D = runner.model # type: ignore    
    dataloader = runner.test_dataloader
    model.eval()
    
    for idx,datas in enumerate(dataloader):
        image = datas['inputs']['img'][0].numpy().transpose((1,2,0)).astype(np.uint8).copy()  # cv2.imread(out.img_path)
        image = cv2.resize(image, (1242,375))
        blank = np.full((400,500,3),255,np.uint8)
        blank.astype(np.uint8).copy()
        outs = model.test_step(datas)
        out = outs[0]
        cam2img:list = out.cam2img
        pred = out.pred_instances_3d
        bboxes:np.ndarray = pred.bboxes_3d.tensor.detach().cpu().numpy()
        labels:np.ndarray = pred.labels_3d.detach().cpu().numpy()
        scores:np.ndarray = pred.scores_3d.detach().cpu().numpy()
        result_image, point = render_result(image, cam2img, bboxes, labels, scores)
        point_image = render_map(blank,point)

        o = cv2.imwrite(os.path.join('work_dirs/', f'mminference_result_{idx}.png'), result_image)
        p = cv2.imwrite(os.path.join('work_dirs/', f'point_inference_{idx}.png'), point_image)
        
        sleep(0.2)
        if idx == 2:
            break
    
if __name__ == '__main__':
    main()
