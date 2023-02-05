import json
import os
import numpy as np
from typing import Any, Dict, List
from mmdet3d.structures.bbox_3d.cam_box3d import CameraInstance3DBoxes

class Asset():
    def __init__(self, path:str = "") -> None:
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

    def load(self, path:str):
        with open(path, 'r') as file:
            asset_dict = json.load(file)

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
        self.model_name = tuple(asset_dict['model_name'])
        self.model_weight = tuple(asset_dict['model_weight'])

        
