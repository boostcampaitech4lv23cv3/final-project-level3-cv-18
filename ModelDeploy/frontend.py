from typing import List
import streamlit as st
import requests
import os
import time
import base64

CONFIG = {
    "title" : "주행 안전 보조 시스템",
    "description" : "초보 운전자 주행 시, 끼어들기나 안전 거리 확보 등을 경고를 통해 안전한 주행에 도움을 주는 시스템",
    "fps" : 5.0,
    "server_url" : "http://localhost:30002",
    "api_asset_list" : "/inference/list",
    "api_load_asset" : "/inference/load",
    "api_video" : "/inference/video",
    "api_image" : "/inference/image",
    "api_map" : "/inference/map",
    "api_status" : "/inference/status"
}

# set state
def init_state_key(key:str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value

init_state_key('is_played', False)
init_state_key('frame_current', 0)
init_state_key('frame_total', 0)
init_state_key('cur_level', "None")
init_state_key('model_status', "Stop")


# set events
def on_play_btn_clicked():
    file = st.session_state.selected_file[1:-1]
    if file == "None":
        print("skip")
        return
    reponse_asset_list = requests.post(CONFIG["server_url"] + CONFIG["api_load_asset"], params={"file":file})
    print(reponse_asset_list.content)
    print(reponse_asset_list.url)
    return

# set page view
st.set_page_config(page_title = CONFIG['title'],
                   layout = "centered",
                   initial_sidebar_state = "expanded"
                )

st.title(CONFIG['title'])
st.text(CONFIG['description'])

# main view
with st.container():
    view_image = st.empty()
    layout_map, layout_control = st.columns(2)
    with layout_map:
        view_map = st.empty()
    with layout_control:
        # file selecter
        reponse_asset_list = requests.get(CONFIG["server_url"] + CONFIG["api_asset_list"])
        content_asset_list = reponse_asset_list.text[1:-1] # type: ignore
        asset_list = [i for i in content_asset_list.split(',')] 
        view_list = st.selectbox('.', asset_list, label_visibility = "collapsed")
        st.session_state.selected_file = view_list
        # play / stop
        if st.session_state.is_played == True:
            button_play = st.button("Stop", on_click=on_play_btn_clicked, type="primary")
        else:
            button_play = st.button("Play", on_click=on_play_btn_clicked, type="secondary")
        level_text = st.text(f'Current Level : {st.session_state.cur_level}')


while True:
    response_image = requests.get(CONFIG["server_url"] + CONFIG["api_image"], stream=True)
    response_map = requests.get(CONFIG["server_url"] + CONFIG["api_map"], stream=True)
    status_temp = requests.get(CONFIG["server_url"] + CONFIG["api_status"], stream=True).json()

    view_image.image(response_image.content)
    view_map.image(response_map.content)
    st.session_state.cur_level = status_temp["cur_level"]
    level_text.write(f'Current Level : {st.session_state.cur_level}')
    
    time.sleep(1./CONFIG["fps"])
