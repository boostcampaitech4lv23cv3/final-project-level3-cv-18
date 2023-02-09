from typing import List, Tuple
import cv2
import numpy as np

class RenderManager:
    """
    ## RenderManager
    Inference 결과들을 이미지에 시각화하는 일을 수행합니다.

    Author : 김형석, 전지용
    """
    COLOR_LEVEL = [
        [  0, 255,   0],
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
        """
        인퍼런스하는 영상이 없을 때 상태를 알려주기 위해 no signal 이미지를 렌더링합니다.
        - image : rendering에 참조 할 이미지(해당 이미지의 shape를 참조하여 동일한 차원으로 rendering 합니다.)
        - fg_color : 글자의 색
        - bg_color : 배경의 색
        """
        h, w, c = image.shape
        text = "No Signal"
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.rectangle(image,(0,0),(w,h),bg_color, thickness=-1)
        textsize = cv2.getTextSize(text, font, 1, 2)[0]
        pos = (int((w - textsize[0])/2), int((h + textsize[1])/2))
        cv2.putText(image, text, pos, font, 1, fg_color)
        return image

        
    def draw_map(self, bboxs:List[np.ndarray], fg_color=(255,0,255), bg_color=(255,255,255)):
        """
        simple한 position map을 렌더링합니다.
        - bboxs : bbox 정보
        - fg_color : graphical context의 색
        - bg_color : 배경의 색
        """
        map_image = np.full((320,320,3), bg_color, np.uint8)
        [cv2.drawMarker(map_image, (bbox[0]+160,bbox[2]+160), fg_color) for bbox in bboxs]
        return map_image

    def alpha_blending(self, blend_target:np.ndarray, source:np.ndarray, alpha:float, mask:np.ndarray) -> None:
        """
        alpha blending을 수행합니다.
        - blend_target : 블렌딩 할 대상 이미지
        - source : 블렌딩 할 이미지
        - alpha : 블렌딩 비율
        - mask : 블렌딩을 수행할 영역
        """
        blend_target[mask > 0] = alpha * blend_target[mask > 0] + (1.-alpha) * source[mask > 0]

    def draw_text(self, image, text, font=cv2.FONT_HERSHEY_SIMPLEX, pos=(0, 0), font_scale=3, font_thickness=2, text_color=(0, 255, 0), text_color_bg=(0, 0, 0)):
        """
        text를 그립니다.
        - image : 글자를 그릴 대상 이미지
        - text : 그릴 글자
        - font : 사용할 폰트
        - pos : 그릴 위치
        - font_scale : 글자의 크기
        - font_thickness : 글자의 굵기
        - text_color : 글자의 색
        - text_color_bg : 글자가 그려지는 영역의 배경색
        """
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        cv2.rectangle(image, pos, (x + text_w, y + text_h+5), text_color_bg, -1)
        cv2.putText(image, text, (x, y + text_h), font, font_scale, text_color, font_thickness)

    def draw_projected_box3d(self, image:np.ndarray, points:np.ndarray, info:List, level=0, thickness=2) -> None:
        """
        Image Plane에 Projection된 3D BBox를 렌더링
        - image : 렌더링할 대상 이미지
        - points : Projection 3D BBox의 corner points
        - info : (distance, rotation)
        - level : 위험 level 수준
        - thickness : graphical context의 굵기
        """
        points = points.astype(np.int32)
        color = RenderManager.COLOR_LEVEL[level]
        
        # Render object info
        distance = info[0]
        rotation = info[1]
        drawpos = [(points[3][0], points[3][1]), (points[3][0], points[3][1]-15)]
        distance_text = f"{distance:.1f}m"
        rotation_text = f"{rotation:.1f}deg"
        self.draw_text(image, distance_text, pos=drawpos[1], font_scale=0.5, font_thickness=2, text_color=color)
        self.draw_text(image, rotation_text, pos=drawpos[0], font_scale=0.5, font_thickness=2, text_color=color)

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
        """
        Bird eyes view 형식의 map을 렌더링합니다.
        - image : 렌더링할 대상 이미지
        - points : Projection 3D BBox들의 corner points
        - levels : 각 Projection 3D BBox에 대응되는 위험 level 값
        """
        (x,y)=(image.shape[1]//2 ,image.shape[0])

        #draw circle
        for idx, r in enumerate(range(50, 500, 100)):
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
        """
        이미지 좌측 상단에 종합적인 위험 상황을 렌더링합니다
        - image : 렌더링할 대상 이미지
        - level : 위험 level 값
        """
        if level == "Warning!": #warning
            color = RenderManager.COLOR_LEVEL[1]
        elif level == 'Danger!!!': #danger
            color = RenderManager.COLOR_LEVEL[2]
        else:
            color = RenderManager.COLOR_LEVEL[0]
        
        self.draw_text(image, level, pos=(0, 0), font_scale=2, font_thickness=3, text_color=color)

        return image