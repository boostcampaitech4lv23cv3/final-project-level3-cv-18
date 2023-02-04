from typing import List, Tuple
import cv2
import numpy as np

class RenderManager:
    COLOR_LEVEL = [
        [255,   0,   0],
        [  0, 255, 255],
        [  0,   0, 255],
    ]
    COLOR_MAP_CIRCLE = [
        [  0,   0, 255],
        [153,   0, 255],
        [  0, 153, 255],
        [  0, 204, 255],
        [153, 255,   0],
        [  0, 255,  51],
        [  0, 255,  51],
        [  0, 255,  51],
        [  0, 255,  51],
        [  0, 255,  51],
        [  0, 255,  51],
        [  0, 255,  51],
        [  0, 255,  51],
        [  0, 255,  51],
    ]
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

    def alpha_blending(self, blend_target:np.ndarray, source:np.ndarray, alpha:float, mask:np.ndarray) -> None:
        blend_target[mask > 0] = alpha * blend_target[mask > 0] + (1.-alpha) * source[mask > 0]

    def draw_projected_box3d(self, image:np.ndarray, points:np.ndarray, level=0, thickness=1) -> None:
        points = points.astype(np.int32)
        color = RenderManager.COLOR_LEVEL[level]
        
        # TODO: 거리표현

        # Render BBox border
        border_image = np.zeros_like(image)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            cv2.line(border_image, (points[i, 0], points[i, 1]), (points[j, 0], points[j, 1]), color, thickness)
            i, j = k + 4, (k + 1) % 4 + 4
            cv2.line(border_image, (points[i, 0], points[i, 1]), (points[j, 0], points[j, 1]), color, thickness)
            i, j = k, k + 4
            cv2.line(border_image, (points[i, 0], points[i, 1]), (points[j, 0], points[j, 1]), color, thickness)
        
        # Render Front Area
        front_image = np.zeros_like(image)
        front = points[[0,3,7,4],:]
        cv2.drawContours(front_image, [front], -1, color, -1)
        
        # Blending
        self.alpha_blending(image, front_image, 0.9, front_image)
        self.alpha_blending(image, border_image, 0.1, border_image)


    def render_map(self, image:np.ndarray, points:List[np.ndarray], levels:List[int]):
        (x,y)=(image.shape[1]//2 ,image.shape[0])

        #draw circle
        for idx, r in enumerate(range(50, 700, 100)):
            cv2.circle(image, (x,y), r, RenderManager.COLOR_MAP_CIRCLE[idx], thickness=1, lineType=cv2.LINE_AA)
        
        #draw rectangle car
        for idx, (point, level) in enumerate(zip(points,levels)):
            color = RenderManager.COLOR_LEVEL[level]
            rectpoints = point.T
            rectpoints = rectpoints * 10
            rectpoints[:,0] = rectpoints[:,0] + x
            rectpoints[:,1] = y -1 * rectpoints[:,1]
            rectpoints_list = rectpoints.astype(np.int32)
            center = np.mean(rectpoints_list, 0).astype(np.int32)
            front = np.mean(rectpoints_list[[0,3],:], 0).astype(np.int32)
            cv2.polylines(image, [rectpoints_list], True, (0,0,0), thickness=1,lineType=cv2.LINE_AA)
            cv2.fillConvexPoly(image, rectpoints_list, color, lineType=cv2.LINE_AA)
            cv2.arrowedLine(image, center, front, (0,0,0), 2, line_type=cv2.LINE_AA)
        return image

    def draw_level(self, image:np.ndarray, level:str) -> np.ndarray:
        # TODO: level 출력
        font = cv2.FONT_HERSHEY_SIMPLEX
        if level == "Warning!": #warning
            color = RenderManager.COLOR_LEVEL[1]
        elif level == 'Danger!!!': #danger
            color = RenderManager.COLOR_LEVEL[2]
        else:
            color = RenderManager.COLOR_LEVEL[0]
        
        cv2.putText(image, level, (0, 50), font, 2, color, 3)
        return image