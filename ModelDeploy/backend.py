from typing import Dict
import torch
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import Response
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.responses import RedirectResponse
import time
import io
import cv2
import os
import numpy as np
import albumentations as A
import albumentations.pytorch.transforms as tf
from . import modules as md
from . import models as M
from . import utils as ut
from pydantic import BaseModel

# uvicorn ModelDeploy.backend:app --port=30002 --host="172.17.0.2"

class InferenceEngine():
    def __init__(self) -> None:
        self.streamer = md.Streamer()
        self.renderer = md.RenderManager()
        self.model = M.MMSmoke('./mmdetection3d/checkpoints/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth')
        # self.model = M.ONNXSmoke('./work_dirs/end2end.onnx')
        self.asset:md.Asset = None # type: ignore
        self.converter:md.CoordinateConverter = None # type: ignore
        self.loader:md.DataLoaderCV = None # type: ignore
        self.level:str = "None"
        self.status:str = "Stop"

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
        self.status = "Running"

        inference_result = self.model.forward(frame, self.asset.meta_data)
        bboxs = ut.create_bbox3d(inference_result)
        pbboxs = ut.project_bbox3ds(self.converter, bboxs)
        levels = ut.check_danger(inference_result)
        self.level = ut.level2str(levels)
        ut.render_pbboxs(frame, self.renderer, pbboxs, levels)
        ut.render_darw_level(frame, self.renderer, self.level)
        result_map = ut.render_map(renderer=self.renderer, bboxs=bboxs)
        self.streamer.frame = frame
        self.streamer.map = result_map

        return True


class Status(BaseModel):
    cur_model_status: str
    cur_level: str

CONFIG = {
    'defalut_selection' : 'None',
    'path_assets' : './assets',
    'filter_assets' : '.json',
}

app = FastAPI()
engine = InferenceEngine()

@app.get("/")
async def home() -> RedirectResponse: # rediect home url -> /docs
    return RedirectResponse('/docs')

@app.get("/inference/list", response_model=List[str])
async def get_asset_list():
    list = [CONFIG['defalut_selection']]
    list += [f for f in os.listdir(CONFIG['path_assets']) if f.endswith(CONFIG['filter_assets'])]
    return list

@app.post("/inference/load", description="asset을 불러옵니다.")
async def load_asset(file:str) -> HTMLResponse:
    path = os.path.join(CONFIG["path_assets"], file)
    if not os.path.exists(path):
        return HTMLResponse(content="No File", status_code=400)
    else:
        engine.set_engine(path=path)
        return HTMLResponse(content="Done", status_code=200)

@app.get("/inference/video", description="inference되는 Video 입니다.")
async def streaming_video() -> StreamingResponse:
    rx = np.random.random() * engine.streamer.frame.shape[1]
    ry = np.random.random() * engine.streamer.frame.shape[0]
    cv2.drawMarker(engine.streamer.frame, (int(rx), int(ry)), (255,0,255))
    return StreamingResponse(engine.streamer.get_stream_video(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/inference/image")
async def get_image() -> StreamingResponse:
    engine.run_engine()
    return StreamingResponse(content=engine.streamer.stream_image, media_type="image/jpg")

@app.get("/inference/map", description="inference되는 Video 입니다.")
async def get_map() -> StreamingResponse:
    return StreamingResponse(content=engine.streamer.stream_map, media_type="image/jpg")

@app.get("/inference/status", description="현재 Model의 상태를 반환", response_model=Status)
async def create_status():
    st = {'cur_model_status': engine.status, 'cur_level': engine.level}
    return JSONResponse(content=jsonable_encoder(st))
