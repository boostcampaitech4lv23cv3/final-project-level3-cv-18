import cv2
import numpy as np
import albumentations as A
from .. import modules as md
from .. import models as M
from typing import Dict
from .. import utils as ut

class InferenceEngine():
    """
    ## InferenceEngine
    Inference Engine입니다.
    set_engine, run_engine 메서드를 사용하여 제어합니다.
    - set_engine : asset 파일을 불러와 인퍼런스에 필요한 인스턴스를 생성하고 설정합니다.
    - run_engine : 현재 설정으로 engine을 실행합니다.

    Examples:
        >>> engine = InferenceEngine()
        >>> engine.set_engine(some_asset_path)
        >>> engine.run_engine()
    Author : 김형석
    """
    def __init__(self) -> None:
        self.streamer = md.Streamer()
        self.renderer = md.RenderManager()
        self.model:M.ModelBase = None # type: ignore
        self.model_dictionary:Dict[str,M.ModelBase] = {}
        self.asset:md.Asset = None # type: ignore
        self.converter:md.CoordinateConverter = None # type: ignore
        self.loader:md.DataLoaderCV = None # type: ignore
        self.level:str = "None"
        self.status:str = "Stop"

    def __generate_model_key(self, asset:md.Asset):
        name = asset.model_name
        weight = asset.model_weight
        width, height = asset.input_size
        return f"{name}::[{width},{height}]::{weight}"

    def __load_model(self, asset:md.Asset) -> M.ModelBase:
        name = asset.model_name
        weight = asset.model_weight
        width, height = asset.input_size
        model = M.model_factory(name, weight, width, height)
        return model
    
    def __get_model(self, asset:md.Asset) -> M.ModelBase:
        key = self.__generate_model_key(asset)
        if not key in self.model_dictionary.keys():
            self.model_dictionary[key] = self.__load_model(asset)
            print(f"New model has been allocted and loaded - {key}")
        else:
            print(f"Pre allocted model has been loaded - {key}")
        return self.model_dictionary[key]

    def set_engine(self, path:str):
        self.asset = md.Asset(path=path)
        self.converter = md.CoordinateConverter(cam2img=np.array(self.asset.cam2img))
        self.loader = md.DataLoaderCV(path=self.asset.target_path)
        self.model = self.__get_model(self.asset)

    def run_engine(self):
        if self.loader == None:
            self.renderer.draw_no_signal(self.streamer.frame)
            self.renderer.draw_no_signal(self.streamer.map)
            self.status = "Stop"
            return False
        elif not (self.loader.is_opened and self.loader.is_progress):
            self.status = "Stop"
            return False
        ret, frame = self.loader.get_frame()
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if ret == False:
            self.status = "Stop"
            return False
        import time # inference 시간 측정
        start = time.time()
        inference_result = self.model.forward(frame, self.asset.meta_data)
        # test_code
        print(f"Session: {time.time() - start:.5f}sec")
        bboxs = ut.create_bbox3d(inference_result)
        infos = ut.return_info(inference_result)
        pbboxs = ut.project_bbox3ds(self.converter, bboxs)
        levels = ut.check_danger(inference_result)
        self.level = ut.level2str(levels)
        ut.render_pbboxs(frame, self.renderer, pbboxs, levels, infos)
        ut.render_darw_level(frame, self.renderer, self.level)
        result_map = ut.render_map(renderer=self.renderer, bboxs=bboxs, levels=levels)
        self.streamer.frame = frame
        self.streamer.map = result_map
        return True