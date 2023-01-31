import json

import os

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
                "ori_shape": [
                    375,
                    1242
                ],
                "pad_shape": [
                    384,
                    1280
                ]
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

    def __apply_asset(self, asset_dict:dict):
        self._asset_dict = asset_dict
        self.target_path = asset_dict['target_path']
        self.cam2img = asset_dict['cam2img']
        self.trans_mat = asset_dict['trans_mat']
        self.ori_shape = asset_dict['ori_shape']
        self.pad_shape = asset_dict['pad_shape']

        
