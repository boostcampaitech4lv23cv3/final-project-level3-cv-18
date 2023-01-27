import numpy as np
import os
from typing import Optional

class CoordinateConverter:
    cam2vel:np.ndarray
    vel2cam:np.ndarray
    cam2img:np.ndarray
    R0:np.ndarray
    """ 
    CoordinateConverter
     - 이 Class는 좌표계 변환을 수행합니다.
     - 작성자 : 김형석
    """

    def __init__(self, cam2vel:Optional[np.ndarray]=None, # type: ignore   
                        vel2cam:Optional[np.ndarray]=None, # type: ignore   
                        cam2img:Optional[np.ndarray]=None, # type: ignore   
                        R0:Optional[np.ndarray]=None, # type: ignore   
                                         ) -> None:      
        if cam2vel != None:
            self.cam2vel = cam2vel
        if vel2cam != None:
            self.vel2cam = vel2cam
        if cam2img != None:
            self.cam2img = cam2img
        if R0 != None:
            self.R0 = R0
    
    @property
    def c_u(self) -> float :
        return self.cam2img[0, 2]
    @property
    def c_v(self) -> float :
        return self.cam2img[1, 2]
    @property
    def f_u(self) -> float :
        return self.cam2img[0, 0] 
    @property
    def f_v(self) -> float :
        return self.cam2img[1, 1] 
    @property
    def b_x(self) -> float :
        return (self.cam2img[0, 3] / -self.f_u)
    @property
    def b_y(self) -> float :
        return (self.cam2img[1, 3] / -self.f_u) 

    def __convert_cart_to_homo(self, pts_3d):
        """ Input: nx3 points in Cartesian
            Oupput: nx4 points in Homogeneous by pending 1
        """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    def project_vel_to_cam(self, pts_3d_velo):
        pts_3d_velo = self.__convert_cart_to_homo(pts_3d_velo)  # nx4
        return np.dot(pts_3d_velo, np.transpose(self.vel2cam))

    def project_cam_to_vel(self, pts_3d_cam):
        pts_3d_cam = self.__convert_cart_to_homo(pts_3d_cam)  # nx4
        return np.dot(pts_3d_cam, np.transpose(self.cam2vel))

    def project_rect_to_ref(self, pts_3d_rect):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))
    
    def project_cam_to_image(self, points_3d:np.ndarray) -> np.ndarray:
        """Project points in camera coordinates to image coordinates.

        Args:
            points_3d (np.ndarray): Points in shape (N, 3)
            proj_mat (np.ndarray):
                Transformation matrix between coordinates.

        Returns:
            np.ndarray: Points in image coordinates,
                with shape [N, 2].
        """
        points_shape = list(points_3d.shape)
        points_shape[-1] = 1 
        points_4 = np.hstack([points_3d, np.ones(points_shape, points_3d.dtype)])#torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)
        point_2d = points_4 @ self.cam2img
        point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]
        return point_2d_res

    def project_cam_to_rect(self, pts_3d_cam):
        """ Input and Output are nx3 points """
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_cam)))

    def project_rect_to_velo(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx3 points in velodyne coord.
        """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_cam_to_vel(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        pts_3d_ref = self.project_vel_to_cam(pts_3d_velo)
        return self.project_cam_to_rect(pts_3d_ref)

    def project_rect_to_image(self, pts_3d_rect):
        """ Input: nx3 points in rect camera coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.__convert_cart_to_homo(pts_3d_rect)
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.cam2img))  # nx3
        pts_2d[:, 0] /= pts_2d[:, 2] # type: ignore
        pts_2d[:, 1] /= pts_2d[:, 2] # type: ignore
        return pts_2d[:, 0:2] # type: ignore
 
    def project_velo_to_image(self, pts_3d_velo):
        """ Input: nx3 points in velodyne coord.
            Output: nx2 points in image2 coord.
        """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)