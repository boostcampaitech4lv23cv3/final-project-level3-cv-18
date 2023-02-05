import numpy as np
from .. import modules as st
from typing import List

def case_1(label, x_pos, z_pos, r): # 끼어들기 차량
    # return case1_Warning: 1, case1_Danger: 2, safe: 0
    zero_pos = np.array([0, 0])
    xz_pos = np.array([x_pos, z_pos])
    distance = np.sqrt(np.sum(np.square(xz_pos-zero_pos)))
    rotation = np.degrees(r) + 90 # 전방 기준 0도
    # print(f"| x: {x_pos:.4f} | d: {distance:.4f} | r: {rotation:.4f}")
    # check algorithm
    if label == 0: # person
        return 0
    if x_pos > -5 and x_pos < -1: # left line
        if rotation >= 7 and rotation <= 30: # car head
            if distance < 25:
                return 2    # return danger
            elif distance < 50:
                return 1    # return warning
    if x_pos < 5 and x_pos > 1: # right line
        if rotation <= -7 and rotation >= -30: # car head
            if distance < 25:
                return 2    # return danger
            elif distance < 50:
                return 1    # return warning
    return 0    # return safe

def case_2(label, x_pos, z_pos, r): # 전방 차량
    # return case1_Warning: 1, case1_Danger: 2, other: 0
    zero_pos = np.array([0, 0])
    xz_pos = np.array([x_pos, z_pos])
    distance = np.sqrt(np.sum(np.square(xz_pos-zero_pos)))
    rotation = np.degrees(r) + 90 # 전방 기준 0도
    # check algorithm
    if x_pos > -1.5 and x_pos < 1.5: # my line
        if rotation >= -7 and rotation <= 7: # car head
            if distance < 25:
                return 2    # return danger
            elif distance < 50:
                return 1    # return warning
    return 0    # return safe

def check_danger(result:st.InferenceResult) -> List[int]: 
    bboxes = result.bboxes # [x, y, z, h, w, l, r]
    labels = result.labels # 0: 'Pedestrian', 1: 'Cyclist', 2: 'Car'
    scores = result.scores
    # checking case
    levels = [max(case_1(labels[i], result.bboxes[i][0], result.bboxes[i][2], result.bboxes[i][6]), case_2(labels[i], result.bboxes[i][0], result.bboxes[i][2], result.bboxes[i][6])) for i in range(len(bboxes))]
    # print(levels)
    return levels

def level2str(levels) -> str:
    if levels != []:
        total_level = max(levels)
    else:
        total_level = 0
    if total_level == 1:
        return 'Warning!'
    elif total_level == 2:
        return 'Danger!!!'
    else:
        return 'Safe'