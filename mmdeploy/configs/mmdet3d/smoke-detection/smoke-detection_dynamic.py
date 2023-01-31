_base_ = ['./smoke-detection_static.py']

onnx_config = dict(
    dynamic_axes={
        'img': {
            0: 'image_batch',
        }
    },
    input_shape=None)
