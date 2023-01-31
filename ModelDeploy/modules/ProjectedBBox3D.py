import numpy as np

class ProjectedBBox3D:
    def __init__(self, projected_bbox3d:np.ndarray) -> None:
        self.raw_points:np.ndarray = projected_bbox3d