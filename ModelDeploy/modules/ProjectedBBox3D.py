import numpy as np

class ProjectedBBox3D:
    """
    ## ProjectedBBox3D
    Image Plane에 Projecting된 결과를 담기 위한 data structure 입니다.
    
    Author : 김형석
    """
    def __init__(self, projected_bbox3d:np.ndarray) -> None:
        self.raw_points:np.ndarray = projected_bbox3d