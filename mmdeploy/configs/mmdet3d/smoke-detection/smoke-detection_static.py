_base_ = ['../../_base_/onnx_config.py']
codebase_config = dict(
    type='mmdet3d', task='SmokeDetection', model_type='end2end')
onnx_config = dict(
    input_names=['img'],
    output_names=['cls_score', 'bbox_pred'])
