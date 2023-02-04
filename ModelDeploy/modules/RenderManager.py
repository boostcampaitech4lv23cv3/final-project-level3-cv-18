from typing import List, Tuple
import cv2
import numpy as np

class RenderManager:
    def __init__(self) -> None:
        pass

    def draw_no_signal(self, image:np.ndarray, fg_color=(255,255,255), bg_color=(126,126,126)):
        h, w, c = image.shape
        text = "No Signal"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(image,(0,0),(w,h),bg_color, thickness=-1)
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        pos = (int((w - textsize[0])/2), int((h + textsize[1])/2))
        cv2.putText(image, text, pos, font, 1, fg_color)
        return image

        
    def draw_map(self, bboxs:List[np.ndarray], fg_color=(255,0,255), bg_color=(255,255,255)):
        """ Draw map
            points : nx3 ndarray
            fg_color : (r,g,b)
            bg_color : (r,g,b)
        """
        map_image = np.full((320,320,3), bg_color, np.uint8)
        [cv2.drawMarker(map_image, (bbox[0]+160,bbox[2]+160), fg_color) for bbox in bboxs]
        return map_image

    def draw_projected_box3d(self, image:np.ndarray, qs:np.ndarray, level=0, thickness=2) -> np.ndarray:
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
        if level == 1: #warning
            color = (0, 255, 255)
        elif level == 2: #danger
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        
        # TODO: 거리표현

        for k in range(0, 4):
            # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
            i, j = k, (k + 1) % 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
            i, j = k, k + 4
            cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
        
        # TODO: 방향 표현
        # for i in [0,1,5,4]:
        #     cv2.drawMarker(image, (qs[i,0],qs[i,1]), (255,0,0))
        return image


    def render_map(self, image:np.ndarray, points:List[np.ndarray], levels:List[int]):
        (x,y)=(image.shape[1]//2 ,image.shape[0])
        #draw circle
        color_spec=[[0,0,255],[153,0,255],[0,153,255],[0,204,255],[153,255,0],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51],[0,255,51]]
        k=0
        for r in range(50, 700, 100):
            cv2.circle(image, (x,y), r, color_spec[k], thickness=3)
            k+=1
        
        #draw rectangle car
        for idx, (point, level) in enumerate(zip(points,levels)):
            if level == 1: #warning
                color = (0, 255, 255)
            elif level == 2: #danger
                color = (0, 0, 255)
            else:
                color = (255, 0, 0)

            rectpoints = point.T
            rectpoints = rectpoints * 10
            rectpoints[:,0] = rectpoints[:,0] + x
            rectpoints[:,1] = y -1 * rectpoints[:,1]
            rectpoints_list = rectpoints.astype(np.int32)
            cv2.polylines(image, [rectpoints_list], True, (0,0,0), thickness=4,lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(image, rectpoints_list, color)
        return image

    def draw_level(self, image:np.ndarray, level:str) -> np.ndarray:
        # TODO: level 출력
        font = cv2.FONT_HERSHEY_SIMPLEX
        if level == "Warning!": #warning
            color = (0, 255, 255)
        elif level == 'Danger!!!': #danger
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)
        
        cv2.putText(image, level, (0, 50), font, 2, color, 3)
        return image