from typing import Dict
from typing import List
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.responses import StreamingResponse
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from starlette.responses import RedirectResponse
import cv2
import os
import numpy as np
from . import modules as md
from pydantic import BaseModel

# uvicorn ModelDeploy.backend:app --port=30002 --host="172.17.0.2"

class Status(BaseModel):
    cur_model_status: str

class Level(BaseModel):
    cur_level: str

CONFIG = {
    'defalut_selection' : 'None',
    'path_assets' : './assets',
    'filter_assets' : '.json',
}

app = FastAPI()
engine = md.InferenceEngine()

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
    st = {'cur_model_status': engine.status}
    return JSONResponse(content=jsonable_encoder(st))

@app.get("/inference/level", description="위험도 Level 반환", response_model=Level)
async def create_level():
    lv = {'cur_level': engine.level}
    return JSONResponse(content=jsonable_encoder(lv))

@app.post("/inference/model_run", description="현재 Model의 상태를 Running으로 변환")
async def model_run():
    engine.status = 'Running'
    return HTMLResponse(content="Done", status_code=200)

@app.post("/inference/model_stop", description="현재 Model의 상태를 Stop으로 변환")
async def model_stop():
    engine.loader = None
    return HTMLResponse(content="Done", status_code=200)

