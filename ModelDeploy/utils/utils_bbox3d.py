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

def render_pbbox(image:np.ndarray, renderer:st.RenderManager, pbbox:st.ProjectedBBox3D) -> None:
    renderer.draw_projected_box3d(image, pbbox.raw_points)

def render_pbboxs(image:np.ndarray, renderer:st.RenderManager, pbboxs:List[st.ProjectedBBox3D]) -> None:
    [render_pbbox(image=image, renderer=renderer, pbbox=pbbox) for pbbox in pbboxs]

def render_map(renderer:st.RenderManager, bboxs:List[st.BoundingBox3D]):
    return renderer.draw_map([np.array(bbox.center, np.int32) for bbox in bboxs])