import numpy as np
import cv2
from .KITTIObject import *


def project_to_image(pts_3d, P):
    """ Project 3d points to image plane.

    Usage: pts_2d = projectToImage(pts_3d, P)
      input: pts_3d: nx3 matrix
             P:      3x4 projection matrix
      output: pts_2d: nx2 matrix

      P(3x4) dot pts_3d_extended(4xn) = projected_pts_2d(3xn)
      => normalize projected_pts_2d(2xn)

      <=> pts_3d_extended(nx4) dot P'(4x3) = projected_pts_2d(nx3)
          => normalize projected_pts_2d(nx2)
    """
    n = pts_3d.shape[0]
    pts_3d_extend = np.hstack((pts_3d, np.ones((n, 1))))
    # print(('pts_3d_extend shape: ', pts_3d_extend.shape))
    pts_2d = np.dot(pts_3d_extend, np.transpose(P))  # nx3
    pts_2d[:, 0] /= pts_2d[:, 2]
    pts_2d[:, 1] /= pts_2d[:, 2]
    return pts_2d[:, 0:2]


def compute_box_3d(target_object, P):
    rotation_metrix = roty(target_object.ry)
    target_object.print_object()
    h = target_object.h
    w = target_object.w
    l = target_object.l

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    
    corners_3d = np.dot(rotation_metrix, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + target_object.t[0]
    corners_3d[1, :] = corners_3d[1, :] + target_object.t[1]
    corners_3d[2, :] = corners_3d[2, :] + target_object.t[2]
    
    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)

    return corners_2d, np.transpose(corners_3d)

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
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        # cv2.line(image, (qs[i,0],qs[i,1]), (qs[j,0],qs[j,1]), color, thickness, cv2.CV_AA)
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image

def show_image_with_boxes(img, objects, calib, show3d=True, depth=None):
    """ Show image with 2D bounding boxes """
    img_result = np.copy(img)  # for 3d bbox
    # img3 = np.copy(img)  # for 3d bbox
    #TODO: change the color of boxes
    print(calib.P)
    for obj in objects:
        box3d_pts_2d, _ = compute_box_3d(obj, calib.P)
        if box3d_pts_2d is None:
            print("something wrong in the 3D box.")
            continue
        temp = np.array(box3d_pts_2d, np.float32)
        bbox = cv2.boxPoints(cv2.minAreaRect(temp))
        bbox = np.int0(bbox)
        img_result = cv2.drawContours(img_result,[bbox],0,(255,255,255),2)
        if obj.label_str == "Car":
            img_result = draw_projected_box3d(img_result, box3d_pts_2d, color=(0, 255, 0))
        elif obj.label_str == "Pedestrian":
            img_result = draw_projected_box3d(img_result, box3d_pts_2d, color=(255, 255, 0))
        elif obj.label_str == "Cyclist":
            img_result = draw_projected_box3d(img_result, box3d_pts_2d, color=(0, 255, 255))
        else:
            img_result = draw_projected_box3d(img_result, box3d_pts_2d, color=(100, 100, 100))
    
    return img_result

def return_bboxes(x, y, z, h, w, l, ry, P):
    rotation_metrix = roty(ry)

    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
    
    corners_3d = np.dot(rotation_metrix, np.vstack([x_corners, y_corners, z_corners]))
    corners_3d[0, :] = corners_3d[0, :] + x
    corners_3d[1, :] = corners_3d[1, :] + y
    corners_3d[2, :] = corners_3d[2, :] + z
    
    # project the 3d bounding box into the image plane
    corners_2d = project_to_image(np.transpose(corners_3d), P)
    # box3d_pts_2d to 2dbbox
    temp = np.array(corners_2d, np.float32)
    bbox = cv2.boxPoints(cv2.minAreaRect(temp))
    min = np.min(bbox, axis=0)
    max = np.max(bbox, axis=0)
    x_min = min[0] 
    y_min = min[1]
    # 좌표가 이미지 밖으로 벗어났을 때 값이 튀는것을 방지 (임시)
    if x_min < 0:
        x_min = 0.0
    if y_min < 0:
        y_min = 0.0
    if x_min > 1920:
        x_min = 1920.0
    if y_min > 1200:
        y_min = 1200.0
    
    x_max = max[0]
    y_max = max[1]
    if x_max < 0:
        x_max = 0.0
    if y_max < 0:
        y_max = 0.0
    if x_max > 1920:
        x_max = 1920.0
    if y_max > 1200:
        y_max = 1200.0
    return x_min, y_min, x_max, y_max # x_min, y_min, x_max, y_max
    