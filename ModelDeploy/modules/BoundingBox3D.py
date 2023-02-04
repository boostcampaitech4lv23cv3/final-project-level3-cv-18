from typing import Tuple
import torch
import numpy as np


class BoundingBox3D:
    def __init__(self, pred_vector:np.ndarray, label:int, score:float) -> None:
        """
        pred_vector: inference result(7x1 or 1x7 or 7)
        """
        
        pv = pred_vector.tolist()
        self.x:float = pv[0] # type: ignore  
        self.y:float = pv[1] # type: ignore  
        self.z:float = pv[2] # type: ignore  
        self.w:float = pv[3] # type: ignore  
        self.h:float = pv[4] # type: ignore  
        self.l:float = pv[5] # type: ignore  
        self.yaw:float = pv[6] # type: ignore  
        self.rotation_metrix:np.ndarray = BoundingBox3D.__create_rotation_matrix(pv[6])
        self.corners:np.ndarray = BoundingBox3D.__create_corners(
            self.x, # type: ignore  
            self.y, # type: ignore  
            self.z, # type: ignore  
            self.w, # type: ignore  
            self.h, # type: ignore  
            self.l, # type: ignore  
            rotation_metrix = self.rotation_metrix
        )
        self.label = label
        self.score = score

    @property
    def center(self) -> np.ndarray:
        return np.array([self.x,self.y,self.z])
    
    @property
    def map_area_rect(self) -> np.ndarray:
        return np.array([self.corners[0,:-4], self.corners[2,:-4]])

    @staticmethod
    def __create_rotation_matrix(t)->np.ndarray:
        """ Rotation about the y-axis. """
        c:float = np.cos(t) # type: ignore        
        s:float = np.sin(t) # type: ignore        
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    
    @staticmethod
    def __create_corners(x:float,y:float,z:float,w:float,h:float,l:float,rotation_metrix:np.ndarray) -> np.ndarray:
        x_corners = [w/2., -w/2., -w/2., w/2., w/2., -w/2., -w/2., w/2.]
        y_corners = [-h, -h, -h, -h, 0, 0, 0, 0]
        z_corners = [l/2., l/2., -l/2., -l/2., l/2., l/2., -l/2., -l/2.]
        corners_3d = np.dot(rotation_metrix, np.vstack([x_corners, y_corners, z_corners])).astype(np.double) # type: ignore
        corners_3d[0, :] = corners_3d[0, :] + x # type: ignore
        corners_3d[1, :] = corners_3d[1, :] + y # type: ignore
        corners_3d[2, :] = corners_3d[2, :] + z  # type: ignore
        return corners_3d
