_base_ = ['./smoke-detection_static.py', '../../_base_/backends/tensorrt.py']
backend_config = dict(
    common_config=dict(max_workspace_size=1 << 30),
    model_inputs=[
        dict(
            input_shapes=dict(
                img=dict(
                        min_shape=[1, 3, 384, 1280],
                        opt_shape=[1, 3, 384, 1280],
                        max_shape=[1, 3, 384, 1280]
                )
            )
        )
    ])
