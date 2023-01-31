from typing import List
import cv2
import numpy as np

class RenderManager:
    def __init__(self) -> None:
        pass
        
    def draw_map(self, bboxs:List[np.ndarray], fg_color=(255,0,255), bg_color=(255,255,255)):
        """ Draw map
            points : nx3 ndarray
            fg_color : (r,g,b)
            bg_color : (r,g,b)
        """
        map_image = np.full((320,320,3), bg_color, np.uint8)
        [cv2.drawMarker(map_image, (bbox[0]+160,bbox[2]+160), fg_color) for bbox in bboxs]
        return map_image

    def draw_projected_box3d(self, image:np.ndarray, qs:np.ndarray, color=(0, 255, 0), thickness=2) -> np.ndarray:
        """ Draw 3d bounding box in image
            image : input image
            qs: projected points(8,3) 
            - array of vertices for the 3d box in following order:
                1 -------- 0
               /|         /|
              2 -------- 3 .
              | |        | |
              . 5 -------- 4
              |/         |/
              6 -------- 7
        """
        qs = qs.astype(np.int32)
        # cv2.drawContours(image, [[qs[0].tolist(),qs[3].tolist(),qs[7].tolist(),qs[4].tolist()]], -1, (255,0,0))
        for k in range(0, 4):
            # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k, k + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        for i in [0,1,5,4]:
            cv2.drawMarker(image, (qs[i,0],qs[i,1]), (255,0,0))
        return image