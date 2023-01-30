import os
import cv2
import numpy as np

class DataLoaderCV:
    def __init__(self, path:str) -> None:
        if path.endswith('.mp4'):
            self.data_type = 'Video'
        elif path.endswith('.png'):
            self.data_type = 'PNG List'
        else:
            raise Exception("Not supported type. (.mp4 | directory)")
        
        self.path = path
        self.__grabber = cv2.VideoCapture(path)
        self.progress = 0

    @property
    def is_progress(self):
        return self.progress != self.frame_count

    @property
    def is_opened(self):
        return self.__grabber.isOpened()

    @property
    def frame_count(self):
        return int(self.__grabber.get(cv2.CAP_PROP_FRAME_COUNT))
    
    @property
    def frame_width(self):
        return int(self.__grabber.get(cv2.CAP_PROP_FRAME_WIDTH))
    
    @property
    def frame_height(self):
        return int(self.__grabber.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    @property
    def fps(self):
        return self.__grabber.get(cv2.CAP_PROP_FPS)
    
    def get_frame(self) -> (bool, np.ndarray): # type: ignore
        self.progress += 1
        ret, frame = self.__grabber.read()
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return ret, image