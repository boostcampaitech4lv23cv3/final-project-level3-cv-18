from ._ModelBase import *
from ._MMSmoke import *
from ._ONNXSmoke import *
# from ._TRTSmoke import *

def model_factory(name:str,
                 weight_path:str,
                 input_width:int,
                 input_height:int
                  ) -> ModelBase:
    model_list = {
        "MMSmoke" : MMSmoke,
        "ONNXSmoke" : ONNXSmoke,
        # "TRTSmoke" : TRTSmoke,
    }
    generator = model_list[name]
    model = generator(weight_path=weight_path, input_width=input_width, input_height=input_height)
    return model