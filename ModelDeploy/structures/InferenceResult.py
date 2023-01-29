from torch import Tensor
import numpy as np

class InferenceResult:
    bboxes:np.ndarray
    labels:np.ndarray
    scores:np.ndarray
    size:int

    def __init__(self, bboxes:Tensor, labels:Tensor, scores:Tensor) -> None:
        self.bboxes:np.ndarray = bboxes.detach().cpu().numpy()
        self.labels:np.ndarray = labels.detach().cpu().numpy()
        self.scores:np.ndarray = scores.detach().cpu().numpy()
        self.size = scores.shape[0]