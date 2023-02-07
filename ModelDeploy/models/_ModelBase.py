from abc import ABCMeta, abstractmethod
from typing import Any, Dict, List, Literal
import numpy as np
from .. import modules as md

__all__ = ['ModelBase']
 
class ModelBase(metaclass=ABCMeta):
    def __init__(self, 
                 weight_path:str,
                 input_width:int,
                 input_height:int, 
                 input_type:Literal["ndarray","tensor"]) -> None:
        self.__input_width = input_width
        self.__input_height = input_height
        self.__input_converter = md.InputConverter(input_width=input_width, 
                                                   input_height=input_height, 
                                                   input_type=input_type)
        self.__weight_path = weight_path
    
    def forward(self, image:np.ndarray, meta_data:List[Dict[str, Any]]) -> md.InferenceResult:
        input_data = self.__input_converter(image)
        return self._forward(input_data, meta_data)

    @abstractmethod
    def _forward(self, input_data:Any, meta_data:List[Dict[str, Any]]) -> md.InferenceResult:
        pass