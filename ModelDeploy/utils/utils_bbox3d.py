from typing import List
from .. import modules as st
import numpy as np



def create_bbox3d(result:st.InferenceResult) -> List[st.BoundingBox3D]:
    bboxes = result.bboxes
    labels = result.labels
    scores = result.scores
    box_list = []
    for box, label, score in zip(bboxes, labels, scores):
        box_list.append(st.BoundingBox3D(pred_vector=box, label=label, score=score))
    return box_list
    
    
def project_bbox3d(converter:st.CoordinateConverter, bbox:st.BoundingBox3D) -> st.ProjectedBBox3D:
    projected_bbox3d = converter.project_cam_to_image(np.transpose(bbox.corners))
    pbbox = st.ProjectedBBox3D(projected_bbox3d=projected_bbox3d)
    return pbbox

def project_bbox3ds(converter:st.CoordinateConverter, bboxs:List[st.BoundingBox3D]) -> List[st.ProjectedBBox3D]:
    return [project_bbox3d(converter, box) for box in bboxs]

def render_pbbox(image:np.ndarray, renderer:st.RenderManager, pbbox:st.ProjectedBBox3D, level:int, info:List) -> None:
    renderer.draw_projected_box3d(image, pbbox.raw_points, info, level=level)

def render_pbboxs(image:np.ndarray, renderer:st.RenderManager, pbboxes:List[st.ProjectedBBox3D], levels:List[int], infos:List) -> None:
    [render_pbbox(image=image, renderer=renderer, pbbox=pbboxes[i], level=levels[i], info=infos[i]) for i in range(len(pbboxes))]

def render_map(renderer:st.RenderManager, bboxs:List[st.BoundingBox3D], levels:List[int]):
    map_image = np.full((500,500,3), (255,255,255), np.uint8)
    return renderer.render_map(image=map_image,
                               points=[bbox.map_area_rect for bbox in bboxs],
                               levels=levels)

def render_darw_level(image:np.ndarray, renderer:st.RenderManager, level_str:str):
    return renderer.draw_level(image, level_str)

def return_info(result:st.InferenceResult) -> List:
    # return object infomation
    # ex) [[distance, rotation, xz_pos], ...]
    bboxes = result.bboxes
    info_list = []
    for box in bboxes:
        zero_pos = np.array([0, 0])
        xz_pos = np.array([box[0], box[2]])
        distance = np.sqrt(np.sum(np.square(xz_pos-zero_pos)))
        rotation = np.degrees(box[6]) + 90 # 전방 기준 0도
        info_list.append([distance, rotation, [box[0], box[2]]])
    return info_list