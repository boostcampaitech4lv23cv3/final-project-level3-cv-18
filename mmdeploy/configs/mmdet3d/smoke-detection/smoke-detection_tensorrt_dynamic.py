_base_ = ['./smoke-detection_dynamic.py', '../../_base_/backends/tensorrt.py']
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                img=dict(
                    img=[3, 384, 1280]
                )
            )
        )
    ])
