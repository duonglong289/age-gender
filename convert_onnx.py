import torch
import torch.nn as nn
import onnx 
import onnxruntime
import numpy as np 

from models.net import ModelAgeGender
import onnxruntime

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()



if __name__ == "__main__":
    model = ModelAgeGender()
    model.init_model("mobilenet_v2", num_age_classes=81)
    estimator = model.load_statedict("weights/orinal_regression_04112020.pt")
    estimator.eval()
    temp_var = torch.rand(10, 3, 224, 224, requires_grad=True)

    onnx_path = "weights/mbnetv2.onnx"

    torch_out = estimator(temp_var)
    torch.onnx.export(estimator,
                    temp_var,
                    onnx_path,
                    export_params=True,
                    opset_version=10,
                    do_constant_folding=True,
                    input_names=['input'],
                    output_names=['output'],
                    dynamic_axes={
                        'input': {0: 'B'},
                        'output': {0: 'B'}
                    })
    
    ort_session = onnxruntime.InferenceSession(onnx_path)

    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(temp_var)}
    ort_outputs = ort_session.run(None, ort_inputs)
    print("Result model onnx: ", ort_outputs[0].shape)
    print("Result model torch: ", len(torch_out))

    np.testing.assert_allclose(to_numpy(torch_out[0]), ort_outputs[0], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[1]), ort_outputs[1], rtol=1e-03, atol=1e-05)
    np.testing.assert_allclose(to_numpy(torch_out[2]), ort_outputs[2], rtol=1e-03, atol=1e-05)
