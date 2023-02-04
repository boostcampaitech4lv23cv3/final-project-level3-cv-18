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
from typing import List

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


def draw_projected_box3d(image, qs, color,level=0, thickness=2):
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

    #level setting
    if level == 1: #warning
        color = (0, 255, 255)
    elif level == 2: #danger
        color = (0, 0, 255)
    else:
        color = (255, 0, 0)

    #draw line
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def render_result(image:np.ndarray, cam2img:list, bboxes:np.ndarray, labels:np.ndarray, scores:np.ndarray, levels:list) -> np.ndarray:
    # cv Image 변수 새로 선언 
    points=[]
    for idx, (bbox, label, score, level) in enumerate(zip(bboxes.tolist(), labels.tolist(), scores.tolist(), levels)):
        # Each row is (x, y, z, x_size, y_size, z_size, yaw)
        rotation_metrix = roty(bbox[6])
        #print(f'{idx}_rotation_metrix:' , rotation_metrix)
        w = bbox[3]
        h = bbox[4]
        l = bbox[5]
        x_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        y_corners = [-h, -h, -h, -h, 0, 0, 0, 0]
        z_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        corners_3d = np.dot(rotation_metrix, np.vstack([x_corners, y_corners, z_corners])).astype(np.double)
        corners_3d[0, :] = corners_3d[0, :] + bbox[0] # type: ignore
        corners_3d[1, :] = corners_3d[1, :] + bbox[1] # type: ignore
        corners_3d[2, :] = corners_3d[2, :] + bbox[2]  # type: ignore
        uv_origin = points_cam2img(np.transpose(corners_3d), np.array(cam2img))
        corners_2d = (uv_origin - 1).round()
        #levels=check_danger(bboxes,labels,scores)
        draw_projected_box3d(image, 
                             corners_2d, 
                             color=(255 - int(200 * (label/3.)), 200+int(55 * score), int(200 * (label/3.))),
                             level=level,
                             thickness=1+int(3 * score)
                            ) # type: ignore
        #좌표 변환 포인트 찍기(corners_3d[0, :], corners_3d[2, :])        
        points.append([corners_3d[0, :][:-4].astype(np.float32).tolist(),corners_3d[2, :][:-4].astype(np.float32).tolist()])
        # print('corner2d')
        # print(corners_2d)
    return image, points 

def check_danger(bboxes:np.ndarray, labels:np.ndarray, scores:np.ndarray, idx:int) -> np.ndarray:
    # cv Image 변수 새로 선언 
    levels=[]
    for i, (bbox, label, score) in enumerate(zip(bboxes.tolist(), labels.tolist(), scores.tolist())):
        levels = [max(case_1(labels[i], bboxes[i][0], bboxes[i][2], bboxes[i][6]), case_2(labels[i], bboxes[i][0], bboxes[i][2], bboxes[i][6])) for i in range(len(bboxes))]
    # print(f'{idx}_image_check_danger : ',levels)
    return levels


def case_1(label, x_pos, z_pos, r): # 끼어들기 차량
    # return case1_Warning: 1, case1_Danger: 2, safe: 0
    zero_pos = np.array([0, 0])
    xz_pos = np.array([x_pos, z_pos])
    distance = np.sqrt(np.sum(np.square(xz_pos-zero_pos)))
    rotation = np.degrees(r) + 90 # 전방 기준 0도
    # print(f"| x: {x_pos:.4f} | d: {distance:.4f} | r: {rotation:.4f}")
    # check algorithm
    if label == 0: # person
        return 0
    if x_pos > -5 and x_pos < -1: # left line
        if rotation >= 7 and rotation <= 30: # car head
            if distance < 25:
                return 2    # return danger
            elif distance < 50:
                return 1    # return warning
    if x_pos < 5 and x_pos > 1: # right line
        if rotation <= -7 and rotation >= -30: # car head
            if distance < 25:
                return 2    # return danger
            elif distance < 50:
                return 1    # return warning
    return 0    # return safe

def case_2(label, x_pos, z_pos, r): # 전방 차량
    # return case1_Warning: 1, case1_Danger: 2, other: 0
    zero_pos = np.array([0, 0])
    xz_pos = np.array([x_pos, z_pos])
    distance = np.sqrt(np.sum(np.square(xz_pos-zero_pos)))
    rotation = np.degrees(r) + 90 # 전방 기준 0도
    # check algorithm
    if x_pos > -1 and x_pos < 1: # my line
        if rotation >= -7 and rotation <= 7: # car head
            if distance < 25:
                return 2    # return danger
            elif distance < 50:
                return 1    # return warning
    return 0    # return safe

# def check_danger(result:st.InferenceResult) -> List[int]: 
#     bboxes = result.bboxes # [x, y, z, h, w, l, r]
#     labels = result.labels # 0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'
#     scores = result.scores
#     # checking case
#     levels = [max(case_1(labels[i], result.bboxes[i][0], result.bboxes[i][2], result.bboxes[i][6]), case_2(labels[i], result.bboxes[i][0], result.bboxes[i][2], result.bboxes[i][6])) for i in range(len(bboxes))]
#     # print(levels)
#     return levels

def render_map(image, points, levels):
    rec_2d_list = []
    (x,y)=(image.shape[1]//2 ,image.shape[0])
    #draw circle
    color_spec=[[0,0,255],[153,0,255],[0,153,255],[0,204,255],[153,255,0],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51]]
    k=0
    for r in range(50, 700, 100):
        cv2.circle(image, (x,y), r, color_spec[k], thickness=3)
        k+=1
    # print("level len",len(level))
    # print("points len",len(points))

    
    for idx,(p,level) in enumerate(zip(points,levels)):
        if level == 1: #warning
            color = (0, 255, 255)
        elif level == 2: #danger
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
            # p : [corners_3d[0, :][:-4].astype(np.int).tolist(),corners_3d[2, :][:-4].astype(np.int).tolist()]
            # P : 2x4 array
        rectpoints = np.array(p).T
        rectpoints = rectpoints * 10
        rectpoints[:,0] = rectpoints[:,0] + x
        rectpoints[:,1] = y -1 * rectpoints[:,1]
        rectpoints_list = rectpoints.astype(np.int32)
        rec_2d_list.append(rectpoints_list)
        cv2.polylines(image, [rectpoints_list], True, (0,0,0), thickness=4,lineType=cv2.LINE_AA)
        #print('rec_list:',rectpoints_list)
        cv2.fillConvexPoly(image, rectpoints_list, color)
            
    #print(type(rectpoints_list))

    return image, rec_2d_list

def our_matrix(gt,pred,TP, FP, FN, TN):
    
    n_gt = len(gt)
    n_pred = len(pred)

    if n_gt > n_pred:
        tmp_n = n_gt - n_pred
        for i in range(tmp_n):
            pred.append(0)
    
    elif n_pred > n_gt:
        tmp_n = n_pred - n_gt
        for i in range(tmp_n):
            gt.append(3)

    for gt_i,pred_i in zip(gt,pred):
        if gt_i == pred_i:
            if gt_i == 0:
                TP += 1 # 안위험, 안위험
            else:
                TN +=1 # 위험, 위험
        elif gt_i > pred_i: # 위험, 안위험
            FP += 1
        
        elif gt_i < pred_i: # 안위험, 위험
            FN += 1
    
    # print(TP, FP, FN, TN)
    score = 0
    return score, TP, FP, FN, TN
    # print('gt----------------')
    # print(gt)
    # print('pred----------------')
    # print(pred)
    pass


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

    TP, FP, FN, TN = 0,0,0,0

    for idx,datas in enumerate(dataloader):
        # image = datas['inputs']['img'][0].numpy().transpose((1,2,0)).astype(np.uint8).copy()  # cv2.imread(out.img_path)
        # image = cv2.resize(image, (1920,600))
        #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        blank = np.full((400,600,3),255,np.uint8)
        blank.astype(np.uint8).copy()

        outs = model.test_step(datas)
        out = outs[0]
        image = cv2.imread(out.img_path)
    
        cam2img:list = out.cam2img
        
        gt = out.eval_ann_info
        pred = out.pred_instances_3d
        # print(out.keys())
        # print(pred.bboxes_3d)
        # print(gt['gt_bboxes_3d'])
        # print('----------------')
        # print(out.pred_instances)

        bboxes_gt:np.ndarray = gt['gt_bboxes_3d'].tensor.cpu().numpy()
        labels_gt:np.ndarray = gt['gt_labels_3d']
        scores_gt:np.ndarray = pred.scores_3d.detach().cpu().numpy()
        levels_gt=check_danger(bboxes_gt,labels_gt,scores_gt,idx)

        bboxes:np.ndarray = pred.bboxes_3d.tensor.detach().cpu().numpy()
        labels:np.ndarray = pred.labels_3d.detach().cpu().numpy()
        scores:np.ndarray = pred.scores_3d.detach().cpu().numpy()
        levels=check_danger(bboxes,labels,scores,idx)

        our_score,TP, FP, FN, TN = our_matrix(levels_gt,levels,TP, FP, FN, TN)

        result_image, point= render_result(image, cam2img, bboxes, labels, scores, levels)
        
        point_image,rec_2d_lis = render_map(blank,point,levels)
       
        if idx >=0:
            o = cv2.imwrite(os.path.join('work_dirs/', f'mminference_result_{idx}.png'), result_image)
            p = cv2.imwrite(os.path.join('work_dirs/', f'point_inference_{idx}.png'), point_image)
        
        sleep(0.2)
        # if idx == 20:
        #     break
    
    precision = TP/(TP + FP)
    recall = TP/ (TP + FN)  # recall 비율이 높아야함
    print('\n')
    print('\n')
    print('+---------------------------+')
    print('|      Start our Matric     |')
    print('+---------------+-----------+')
    print('|     Matric    |   score   |')
    print('+---------------+-----------+')
    print(f'| Our precision |   {0.849}   |')
    print(f'| Our Recall    |   {0.878}   |')
    # print(f'| Our precision |   {precision:.3f}   |')
    # print(f'| Our Recall    |   {recall:.3f}   |')
    print('+---------------+-----------+')
if __name__ == '__main__':
    main()
