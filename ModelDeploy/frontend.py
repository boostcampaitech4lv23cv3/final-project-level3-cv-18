from typing import List
import streamlit as st
import requests
import os
import time
import base64

CONFIG = {
    "title" : "주행 안전 보조 시스템 (Web Demo)",
    "description" : "초보 운전 시, 위험 상황 경고를 통해 안전한 주행에 도움을 주는 시스템",
    "fps" : 30.0,
    "server_url" : "http://localhost:30002",
    "api_asset_list" : "/inference/list",
    "api_load_asset" : "/inference/load",
    "api_video" : "/inference/video",
    "api_image" : "/inference/image",
    "api_map" : "/inference/map",
    "api_status" : "/inference/status",
    "api_level" : "/inference/level",
    "api_model_run" : "/inference/model_run",
    "api_model_stop" : "/inference/model_stop"
}

# set state
def init_state_key(key:str, value) -> None:
    if key not in st.session_state:
        st.session_state[key] = value

init_state_key('is_played', False)
init_state_key('frame_current', 0)
init_state_key('frame_total', 0)
init_state_key('model_status', "Stop")

# set events
def on_play_btn_clicked():
    print("button")
    if st.session_state.is_played: # Playing
        st.session_state.is_played = False
        requests.post(CONFIG["server_url"] + CONFIG["api_model_stop"])
        return
    else:
        file = st.session_state.selected_file[1:-1]
        if file == "None":
            print("skip")
            requests.post(CONFIG["server_url"] + CONFIG["api_model_stop"])
            return
        st.session_state.is_played = True
        requests.post(CONFIG["server_url"] + CONFIG["api_model_run"])
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
    layout_map, layout_control = st.columns([1, 1.618])
    with layout_map:
        st.subheader("Bird Eyes View")
        view_map = st.empty()
    with layout_control:
        st.subheader("Controller")
        # file selecter
        reponse_asset_list = requests.get(CONFIG["server_url"] + CONFIG["api_asset_list"])
        content_asset_list = reponse_asset_list.text[1:-1] # type: ignore
        asset_list = [i for i in content_asset_list.split(',')] 
        view_list = st.selectbox('.', asset_list, label_visibility = "collapsed")
        st.session_state.selected_file = view_list
        # play / stop
        if st.session_state.is_played == True:
            button_play = st.button("Stop", on_click=on_play_btn_clicked, type="secondary") # secondary
        else:
            button_play = st.button("Play", on_click=on_play_btn_clicked, type="secondary")  # primary
        st.subheader("Status")
        level_text = st.text(f'Current Level : None')
        Model_text = st.text(f'Current Model Status : {st.session_state.model_status}')

first = True
# Main loop
while True:
    if first:
        response_image = requests.get(CONFIG["server_url"] + CONFIG["api_image"], stream=True)
        response_map = requests.get(CONFIG["server_url"] + CONFIG["api_map"], stream=True)
        view_image.image(response_image.content)
        view_map.image(response_map.content)
        first = False
    
    status_temp = requests.get(CONFIG["server_url"] + CONFIG["api_status"], stream=True).json()
    st.session_state.model_status = status_temp["cur_model_status"]

    Model_text.write(f'Current Model Status : {st.session_state.model_status}')

    if st.session_state.model_status == "Running":
        response_image = requests.get(CONFIG["server_url"] + CONFIG["api_image"], stream=True)
        level_temp = requests.get(CONFIG["server_url"] + CONFIG["api_level"], stream=True).json()  # for level sync
        response_map = requests.get(CONFIG["server_url"] + CONFIG["api_map"], stream=True)
        view_image.image(response_image.content)
        view_map.image(response_map.content)

        cur_level = level_temp["cur_level"]
        level_text.write(f'Current Level : {cur_level}')
        
    time.sleep(1./CONFIG["fps"])
