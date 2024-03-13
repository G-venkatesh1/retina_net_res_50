

import onnxruntime as ort
# from utils import timetaken
from onnxruntime.quantization.calibrate import (CalibrationDataReader,
                                                CalibrationMethod)
from onnxruntime.quantization.quantize import quantize_static, quantize_dynamic
from onnxruntime.quantization.preprocess import quant_pre_process
from onnxruntime.quantization.quant_utils import QuantType,QuantFormat

class OnnxStaticQuantization:
    def __init__(self) -> None:
        self.enum_data = None
        self.calibration_technique = {
            "MinMax": ort.quantization.calibrate.CalibrationMethod.MinMax,
            "Entropy": ort.quantization.calibrate.CalibrationMethod.Entropy,
            "Percentile": ort.quantization.calibrate.CalibrationMethod.Percentile,
            "Distribution": ort.quantization.calibrate.CalibrationMethod.Distribution
        }

    def get_next(self, EP_list = ['CPUExecutionProvider']):
        if self.enum_data is None:
            session = ort.InferenceSession(self.fp32_onnx_path, providers=EP_list)
            input_name = session.get_inputs()[0].name
            calib_list = []
            count = 0
            for nhwc_data in range(len(self.calibration_loader)):
                image = self.calibration_loader[nhwc_data]
                # print(image['img'].shape)
                calib_list.append({input_name: image['img'].unsqueeze(0).permute(0,3,2,1).numpy()}) 
                if self.sample == count: break
                count = count + 1
            self.enum_data = iter(calib_list)
        return next(self.enum_data, None)
    
    # @timetaken
    def quantization(self, fp32_onnx_path, future_int8_onnx_path, calib_method, calibration_loader, sample=100):
        
        self.sample = sample
        self.calibration_loader = calibration_loader 
        
        _ = quantize_static(
                model_input=fp32_onnx_path,
                model_output=future_int8_onnx_path,
                calibrate_method=self.calibration_technique[calib_method],
                quant_format=QuantFormat.QDQ,
                weight_type=QuantType.QInt16,
                activation_type=QuantType.QInt16,
                per_channel=True, reduce_range=True,
                calibration_data_reader=self
            )
        return self