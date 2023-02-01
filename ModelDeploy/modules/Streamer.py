import io
import numpy as np
import cv2

class Streamer():
    def __init__(self) -> None:
        self.frame:np.ndarray = np.full((320,1280,3), 255, np.uint8)
        self.map:np.ndarray = np.full((320,320,3), 255, np.uint8)

    def get_stream_video(self):
        while True:
            ret, buffer = cv2.imencode('.jpg', self.frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(frame) + b'\r\n')
    

    def get_stream_map(self):
        while True:
            ret, buffer = cv2.imencode('.jpg', self.map)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(frame) + b'\r\n')
            
    @property
    def stream_image(self):
        ret, buffer = cv2.imencode('.jpg', self.frame)
        frame = buffer.tobytes()
        image_stream = io.BytesIO(buffer)
        return image_stream
    
    @property
    def stream_map(self):
        ret, buffer = cv2.imencode('.jpg', self.map)
        frame = buffer.tobytes()
        image_stream = io.BytesIO(buffer)
        return image_stream
