import cv2
import numpy as np
import albumentations as A
from .. import modules as md
from .. import models as M
from .. import utils as ut

class InferenceEngine():
    def __init__(self) -> None:
        self.streamer = md.Streamer()
        self.renderer = md.RenderManager()
        self.model = M.TRTSmoke('./ModelDeploy/models/smoke_trt.engine')
        self.asset:md.Asset = None # type: ignore
        self.converter:md.CoordinateConverter = None # type: ignore
        self.loader:md.DataLoaderCV = None # type: ignore
        self.level:str = "None"
        self.status:str = "Stop"

    def __load_model(self, name:str, weight_path:str):

        return 0

    def set_engine(self, path:str):
        self.asset = md.Asset(path=path)
        self.converter = md.CoordinateConverter(cam2img=np.array(self.asset.cam2img))
        self.loader = md.DataLoaderCV(path=self.asset.target_path)

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
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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