from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List
import numpy as np
from .. import modules as md

 
class ModelBase(metaclass=ABCMeta):
    
    @abstractmethod
    def forward(self, image:np.ndarray, meta_data:List[Dict[str, Any]]) -> md.InferenceResult:
        pass