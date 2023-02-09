from torch import Tensor
import numpy as np

class InferenceResult:
    """
    ## InferenceResult
    Inference 결과를 담기 위한 data structure 입니다.
    
    Author : 김형석
    """
    bboxes:np.ndarray
    labels:np.ndarray
    scores:np.ndarray
    size:int

    def __init__(self, bboxes:Tensor, labels:Tensor, scores:Tensor) -> None:
        self.bboxes:np.ndarray = bboxes.detach().cpu().numpy()
        self.labels:np.ndarray = labels.detach().cpu().numpy()
        self.scores:np.ndarray = scores.detach().cpu().numpy()
        self.size = scores.shape[0]