from typing import List
import structures as st
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


