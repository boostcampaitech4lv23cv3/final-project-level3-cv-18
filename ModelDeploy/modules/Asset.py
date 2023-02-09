import json
import os
import numpy as np
from typing import Any, Dict, List, Tuple
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes
import cv2

class Asset():
    """
    ## Asset
    Runtime에 Inference Engine 동작에 필요한 meta data들을 보유합니다.
    asset은 json으로 serialize 되어있으며 instane 할당 시 serialize 된 파일을 불러옵니다.

    Examples:
        >>> asset = Asset(some_path)
    Author : 김형석
    """
    def __init__(self, path:str = "", down_ratio=4) -> None:
        self.down_ratio = down_ratio
        if os.path.exists(path):
            self.load(path)
        else:
            asset_dict ={
                "target_path": "./mmdetection3d/data/kitti/testing/image_2/%06d.png",
                "cam2img": [
                    [
                        721.5377,
                        0.0,
                        609.5593,
                        44.85728
                    ],
                    [
                        0.0,
                        721.5377,
                        172.854,
                        0.2163791
                    ],
                    [
                        0.0,
                        0.0,
                        1.0,
                        0.002745884
                    ],
                    [
                        0.0,
                        0.0,
                        0.0,
                        1.0
                    ]
                ],
                "trans_mat": [
                    [
                        2.5764894e-01,
                        -0.0000000e+00,
                        0.0000000e+00
                    ],
                    [
                        -2.2883824e-17,
                        2.5764894e-01,
                        -3.0917874e-01
                    ],
                    [
                        0.0000000e+00,
                        0.0000000e+00,
                        1.0000000e+00
                    ]
                ],
                "original_size": [
                    1242,
                    375
                ],
                "input_size": [
                    1280,
                    384
                ],
                "model_name" : "MMSmoke",
                "model_weight" : "mmdetection3d/checkpoints/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth"
            }
            self.input_path = path
            self.abs_path = path
            self.__apply_asset(asset_dict=asset_dict)
    
    
    def get_ref_point(self, ref_point1: np.ndarray, ref_point2: np.ndarray) -> np.ndarray:
        """Get reference point to calculate affine transform matrix.
        While using opencv to calculate the affine matrix, we need at least
        three corresponding points separately on original image and target
        image. Here we use two points to get the the third reference point.
        """
        d = ref_point1 - ref_point2
        ref_point3 = ref_point2 + np.array([-d[1], d[0]])
        return ref_point3

    # calc trans_mat
    def get_transform_matrix(self, center:Tuple, scale:Tuple, output_scale:Tuple[float]) -> np.ndarray:
        """
        Get affine transform maxtirx.
        Args:
            center: Center of current image
            scale: Scale of current image
            output_scale: The transform target image scales
        """
        src_w = scale[0]
        dst_w = output_scale[0]
        dst_h = output_scale[1]
        src_dir = np.array([0, src_w * -0.5])
        dst_dir = np.array([0, dst_w * -0.5])
        src = np.zeros((3, 2), dtype=np.float32)
        dst = np.zeros((3, 2), dtype=np.float32)
        src[0, :] = center
        src[1, :] = center + src_dir
        dst[0, :] = np.array([dst_w * 0.5, dst_h * 0.5])
        dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir
        src[2, :] = self.get_ref_point(src[0, :], src[1, :])
        dst[2, :] = self.get_ref_point(dst[0, :], dst[1, :])
        get_matrix = cv2.getAffineTransform(src, dst)
        matrix = np.concatenate((get_matrix, [[0., 0., 1.]]))
        return matrix.astype(np.float32)

    def load(self, path:str):
        with open(path, 'r') as file:
            asset_dict = json.load(file)
        if not "trans_mat" in asset_dict.keys():
            asset_dict["trans_mat"] = self.get_transform_matrix(
                    (asset_dict["original_size"][0]/2, asset_dict["original_size"][1]/2), 
                    (asset_dict["original_size"][0], asset_dict["original_size"][1]),
                    (asset_dict["input_size"][0]//self.down_ratio, asset_dict["input_size"][1]//self.down_ratio))
        print(asset_dict)
        self.input_path = path
        self.abs_path = os.path.abspath(path)
        self.__apply_asset(asset_dict=asset_dict)
    
    @property
    def meta_data(self) -> List[Dict[str, Any]]:
        meta_data = [
            {
                "cam2img": self.cam2img,
                "trans_mat": np.array(self.trans_mat),
                "ori_shape": (self.original_size[1],self.original_size[0]),
                "pad_shape": (self.input_size[1],self.input_size[0]),
                "box_type_3d": CameraInstance3DBoxes
            }
        ]
        return meta_data

    def __apply_asset(self, asset_dict:dict):
        self._asset_dict = asset_dict
        self.target_path = asset_dict['target_path']
        self.cam2img = list(asset_dict['cam2img'])
        self.trans_mat = list(asset_dict['trans_mat'])
        self.original_size = tuple(asset_dict['original_size'])
        self.input_size = tuple(asset_dict['input_size'])
        self.model_name = asset_dict['model_name']
        self.model_weight = asset_dict['model_weight']

        
