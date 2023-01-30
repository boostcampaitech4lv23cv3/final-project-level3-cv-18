from typing import List
import streamlit as st
from random import randint
import requests
import cv2
import numpy as np
import os
import time


CONFIG = {
    'defalut_selection' : 'None',
    'path_assets' : './assets',
    'filter_assets' : '.mp4',
    'fps' : 30.0,
}

# set state
def init_state_key(key:str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value

def get_files() -> List[str]:
    list = [CONFIG['defalut_selection']]
    list += [f for f in os.listdir(CONFIG['path_assets']) if f.endswith(CONFIG['filter_assets'])]
    return list

def alloc_loader(path:str):
    st.session_state.image_loader = cv2.VideoCapture(path)

def release_loader():
    st.session_state.image_loader = None

init_state_key('title', "주행 안전 보조 시스템")
init_state_key('description', "초보 운전자가 도로를 주행할 때 끼어들기나 안전 거리 확보 등을 경고를 통해 안전한 주행에 도움을 주는 시스템")
init_state_key('update_count', 0)
init_state_key('selected_file', CONFIG['defalut_selection'])
init_state_key('is_played', False)
init_state_key('enable_inference', True)
init_state_key('image_loader', None)

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
        view_map = st.image(np.zeros((480,640,3)))
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
        cap:cv2.VideoCapture = st.session_state.image_loader
        if cap.isOpened():
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            view_image.image(frame)
            view_map.image(cv2.resize(frame, (640,480)))

    st.session_state.update_count += 1
    view_status.text(f"Update:{st.session_state.update_count}|\
                     Play:{st.session_state.is_played}|\
                     Inference:{st.session_state.enable_inference}\
                     ")
    time.sleep(1./CONFIG['fps'])

