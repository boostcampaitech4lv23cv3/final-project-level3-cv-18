# TODO: Make Function Tools

### -> single_stage_mono3.py:92
def extract_feat(self, batch_inputs_dict: dict) -> Tuple[Tensor]:
        ####
        ####
        ####

        print(f'START: InputSize[{batch_imgs.size()}]')

        print(self.backbone)

        import torch
        device = 'cuda:0'
        dummy_input = torch.randn(1, 3, 384, 1280, requires_grad=True, device=device)
        self.backbone.eval()

        # Jit Trace
        #tt = torch.jit.trace(self.backbone, dummy_input)

        # Jit Script
        #ts = torch.jit.script(self.backbone)

        # Torch-TensorRT
        import torch_tensorrt
        enabled_precisions = {torch.float}
        trt_ts_module = torch_tensorrt.compile(
            self.backbone, inputs=dummy_input, enabled_precisions=enabled_precisions
        )
        out2 = trt_ts_module(dummy_input)
        print(type(out2))


        # ONNX
        torch.onnx.export(self.backbone, dummy_input, "/home/admin/detection3d/mmdetection3d/test.onnx",
                          verbose=True,
                          export_params=True,
                          opset_version=17,
                          do_constant_folding=True,
                          input_names=['input'],
                          output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'},
                                        'output': {0: 'batch_size'}})

        import numpy as np
        import onnxruntime as ort
        ort_session = ort.InferenceSession("/home/admin/detection3d/mmdetection3d/test.onnx", providers=['CUDAExecutionProvider'])

        def to_numpy(tensor):
            return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

        ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(dummy_input)}
        ort_outs = ort_session.run(None, ort_inputs)

        # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
        out1 = tuple(t.cpu() for t in self.backbone(dummy_input))
        np.testing.assert_allclose(to_numpy(out1[0]), ort_outs[0], rtol=1e-03, atol=1e-05)
        np.testing.assert_allclose(to_numpy(out1[0]), out2[0], rtol=1e-03, atol=1e-05)
        print("Exported model has been tested with ONNXRuntime, and the result looks good!")

        #m = torch.jit.script(self.backbone)
        #torch.jit.save(m, '/home/admin/detection3d/mmdetection3d/test.pth')

        if self.with_neck:
            x = self.neck(x)
        for xxx in x:
            print(type(xxx))
            print(xxx)
        return x
