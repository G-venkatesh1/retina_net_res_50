import torch
from torchvision import transforms
import argparse
from retinanet import model
from retinanet import coco_eval
from retinanet.dataloader import CocoDataset, Resizer, Normalizer,Resizer_const
import onnxruntime as ort
from retinanet import model,quantise
import onnx
import tvm
from tvm import relay
import numpy as np
from onnxconverter_common import float16
def main(args=None):
    #load dataset
    dataset_val = CocoDataset('/kaggle/input/coco-2017-dataset/coco2017', set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer_const()]))
    #load fp32_model
    retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet.load_state_dict(torch.load("/kaggle/input/retina_net_resnet-50/other/retina_net_model/1/coco_resnet_50_map_0_335_state_dict.pt"))
        # retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet.load_state_dict(torch.load("/kaggle/input/retina_net_resnet-50/other/retina_net_model/1/coco_resnet_50_map_0_335_state_dict.pt"))
        # retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = False
    retinanet.eval()
    retinanet.freeze_bn()
    method="fp_32_baseline"
    pr=32
    # coco_eval.evaluate_coco(dataset_val,retinanet,method,pr)
    #fp32_to_onnx
    example_input = torch.randn(1, 3,640,640).cuda()
    onnx_fp_32_path = "/kaggle/working/fp_32.onnx"
    torch.onnx.export(retinanet,example_input,onnx_fp_32_path,opset_version=15)
    ort_session = onnxruntime.InferenceSession('/kaggle/working/fp_32.onnx')
    method="onnx"
    coco_eval.evaluate_coco(dataset_val,ort_session,method,pr)
    # #fp32_to_fp16_onnx
    # onnx_fp_16_path ="/home/ubuntu/workspace/Venkatesh/retina_net_res_50/fp_16.onnx"
    # onnx_32_model = onnx.load(onnx_fp_32_path)
    # onnx_16_model = float16.convert_float_to_float16(onnx_32_model,min_positive_val=1e-7,max_finite_val=1e4)
    # onnx.save(onnx_16_model,onnx_fp_16_path)
    # ort_16_session = onnxruntime.InferenceSession("/home/ubuntu/workspace/Venkatesh/retina_net_res_50/fp_32.onnx")
    # pr=16
    # coco_eval.evaluate_coco(dataset_val,ort_16_session,method,pr)
    # #int8
    int8_onnx_path ="/home/ubuntu/workspace/Venkatesh/retina_net_res_50/int_w8a8_mm_500_n.onnx"
    pre_processed_path ="/home/ubuntu/workspace/Venkatesh/retina_net_res_50/fp_32_pre_processed.onnx"
    ort.quantization.shape_inference.quant_pre_process(onnx_fp_32_path, pre_processed_path)
    module = quantise.OnnxStaticQuantization()
    module.fp32_onnx_path =pre_processed_path
    module.quantization(
        fp32_onnx_path=pre_processed_path,
        future_int8_onnx_path=int8_onnx_path,
        calib_method="MinMax",
        calibration_loader=dataset_val,
        sample=500
    )
    # onnx_model_path = '/kaggle/input/int_8_onnx_model/onnx/retina_net/1/400_int_8_mm_w8a8.onnx'
    # onnx_model = onnx.load(onnx_model_path)
    # input_shape = (1,3,640,640)
    # input_name = "input.1"
    # shape_dict ={input_name:input_shape}
    # mod,params = relay.frontend.from_onnx(onnx_model,shape_dict)
    # target = "llvm"
    # method="tvm"
    # with tvm.transform.PassContext(opt_level=3):
    #     executor = relay.build_module.create_executor("graph",mod,tvm.cpu(0),target,params).evaluate()
    # coco_eval.evaluate_coco(dataset_val,executor,method,pr)
 
    
    
    
    

























if __name__ == '__main__':
    main()