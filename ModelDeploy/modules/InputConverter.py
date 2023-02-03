from typing import Any, Literal, Tuple
import albumentations as A
import albumentations.pytorch.transforms as tf
import torch
import numpy as np

class InputConverter:
    """
    ## InputConverter
    - Convert image to input data format
    
    작성자 : 김형석
    """
    def __init__(self, 
                 input_width:int, 
                 input_height:int, 
                 input_type:Literal["ndarray","tensor"],
                 mean:Tuple[float,float,float]=(0.485, 0.456, 0.406), 
                 std:Tuple[float,float,float]=(0.229, 0.224, 0.225)) -> None:
        self._input_type = input_type
        self._transform = A.Compose([A.Resize(input_height,input_width),A.Normalize(mean,std)])
        self._totensor = tf.ToTensorV2()
        self._converter = self._convert_to_ndarray_input if input_type == "ndarray" else self._convert_to_tensor_input

    def __call__(self, image:np.ndarray) -> Any:
        image = self._transform(image = image)['image']
        input_data = self._converter(image)
        return input_data # type: ignore

    def _convert_to_ndarray_input(self, image:np.ndarray) -> np.ndarray:
        ndarray = image.transpose((2, 0, 1))  
        ndarray = np.expand_dims(ndarray, axis=0)
        return ndarray

    def _convert_to_tensor_input(self, image:np.ndarray) -> torch.Tensor:
        tensor = self._totensor(image=image)['image']
        tensor = tensor.unsqueeze(dim=0)
        return tensor