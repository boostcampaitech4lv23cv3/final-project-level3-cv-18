from typing import List
import streamlit as st
from random import randint
import requests
import cv2
import numpy as np
import os
import time
import utils as ut
import models as M
import modules as md
import albumentations as A
import albumentations.pytorch.transforms as tf


CONFIG = {
    'defalut_selection' : 'None',
    'path_assets' : './assets',
    'filter_assets' : '.json',
    'fps' : 30.0,
}

# @st.cache
def create_model():
    model = M.MMSmoke('./mmdetection3d/checkpoints/smoke/smoke_dla34_pytorch_dlaneck_gn-all_8x4_6x_kitti-mono3d_20210929_015553-d46d9bb0.pth')
    return model

# @st.cache
def create_renderer():
    renderer = md.RenderManager()
    return renderer

# @st.cache
def create_transform():
    return A.Compose([
        A.Resize(384,1280),
        A.Normalize(),
        tf.ToTensorV2(),
    ])

# @st.cache
def create_converter(cam2img:np.ndarray):
    converter = md.CoordinateConverter(cam2img=cam2img)
    return converter

# set state
def init_state_key(key:str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value

def get_files() -> List[str]:
    list = [CONFIG['defalut_selection']]
    list += [f for f in os.listdir(CONFIG['path_assets']) if f.endswith(CONFIG['filter_assets'])]
    return list

def alloc_loader(path:str):
    asset = md.Asset(path)
    image_loader = md.DataLoaderCV(asset.target_path)
    coordinate_converter = md.CoordinateConverter(cam2img=np.array(asset.cam2img))

    st.session_state.model = create_model()
    st.session_state.transform = create_transform()
    st.session_state.renderer = create_renderer()
    st.session_state.asset = asset 
    st.session_state.image_loader = image_loader
    st.session_state.coordinate_converter = coordinate_converter

def release_loader():
    st.session_state.model = None
    st.session_state.transform = None
    st.session_state.renderer = None
    st.session_state.asset = None
    st.session_state.image_loader = None
    st.session_state.coordinate_converter = None

init_state_key('title', "주행 안전 보조 시스템")
init_state_key('description', "초보 운전자 주행 시, 끼어들기나 안전 거리 확보 등을 경고를 통해 안전한 주행에 도움을 주는 시스템")
init_state_key('update_count', 0)
init_state_key('frame_current', 0)
init_state_key('frame_total', 0)
init_state_key('selected_file', CONFIG['defalut_selection'])
init_state_key('is_played', False)
init_state_key('enable_inference', True)
init_state_key('asset', None)
init_state_key('image_loader', None)
init_state_key('coordinate_converter', None)
init_state_key('model', None)
init_state_key('transform', None)
init_state_key('renderer', None)

# set events
def on_play_btn_clicked():
    if st.session_state.is_played == True:
        st.session_state.is_played = False
        release_loader()
        print('stop')
        return

    if st.session_state.selected_file == CONFIG['defalut_selection']:
        print(f'ignore run, selected file is ' + CONFIG['defalut_selection'])
        return
    
    st.session_state.is_played = True
    path = os.path.join(CONFIG['path_assets'], st.session_state.selected_file)
    alloc_loader(path)
    print('play - ', path)
    return


# set page view
st.set_page_config(page_title = st.session_state['title'],
                   layout = "centered",
                   initial_sidebar_state = "expanded"
                )

st.title(st.session_state['title'])
st.text(st.session_state['description'])

# main view
with st.container():
    # display
    view_image = st.image(np.zeros((384,1280,3)))
    # status bar
    view_status = st.empty()
    # 1:1 columns
    layout_map, layout_control = st.columns(2)
    with layout_map:
        # map
        view_map = st.image(np.zeros((320,320,3)))
    with layout_control:
        # file selecter
        view_list = st.selectbox('.', get_files(), label_visibility = "collapsed")
        st.session_state.selected_file = view_list
        checkbox_enable_inference = st.checkbox("Enable Inference")
        st.session_state.enable_inference = checkbox_enable_inference
        # play / stop
        if st.session_state.is_played == True:
            button_play = st.button("Stop", on_click=on_play_btn_clicked, type="primary")
        else:
            button_play = st.button("Play", on_click=on_play_btn_clicked, type="secondary")

        st.spinner('Play & Inference....')


while True:
    if st.session_state.image_loader != None:
        loader:md.DataLoaderCV = st.session_state.image_loader
        model:M.MMSmoke = st.session_state.model
        coordinate_converter:md.CoordinateConverter = st.session_state.coordinate_converter
        transform = st.session_state.transform
        renderer:md.RenderManager = st.session_state.renderer

        if loader.is_opened and loader.is_progress:
            ret, frame = loader.get_frame()
            frame = cv2.resize(frame, (1242,375))
            if ret == False:
                continue
            input_data = transform(image=frame)['image']
            input_data = input_data.to('cuda')
            inference_result = model.forward(input_data)
            bboxs = ut.create_bbox3d(inference_result)
            pbboxs = ut.project_bbox3ds(coordinate_converter, bboxs)
            ut.render_pbboxs(frame, renderer, pbboxs)
            result_map = ut.render_map(renderer=renderer, bboxs=bboxs)
            view_image.image(frame)
            view_map.image(result_map)
            st.session_state.frame_current = loader.progress
            st.session_state.frame_total = loader.frame_count

    st.session_state.update_count += 1
    view_status.text((f"Update:{st.session_state.update_count}|" +
                     f"Play:{st.session_state.is_played}|" +
                     f"Inference:{st.session_state.enable_inference}|" +
                     f"Frame:{st.session_state.frame_current}/{st.session_state.frame_total}|")
                     )
    time.sleep(1./CONFIG['fps'])

