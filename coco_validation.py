import argparse
import torch
from torchvision import transforms
import onnxruntime
import onnx
from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer,Resizer_const
from retinanet import coco_eval
import tvm
from tvm import relay
import numpy as np
# assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))


def func(model_in,method,pr):

    # dataset_val = CocoDataset("/home/ubuntu/Datasets/COCO/coco/", set_name='val2017',
    #                           transform=transforms.Compose([Normalizer(), Resizer_const()]))

    # Create the model
    # method = "fp_32_baseline" #onnx,tvm
    # pr =32#16,8
    if(method=="fp_32_baseline"):
        # retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
        # use_gpu = True

        # if use_gpu:
        #     if torch.cuda.is_available():
        #      retinanet = retinanet.cuda()

        # if torch.cuda.is_available():
        #     retinanet.load_state_dict(torch.load("/home/ubuntu/workspace/Venkatesh/retina_net_res_50/resnet50-19c8e357.pth"))
        #     retinanet = torch.nn.DataParallel(retinanet).cuda()
        # else:
        #     retinanet.load_state_dict(torch.load("/home/ubuntu/workspace/Venkatesh/retina_net_res_50/resnet50-19c8e357.pth"))
        #     retinanet = torch.nn.DataParallel(retinanet)

        model_in.training = False
        model_in.eval()
        model_in.freeze_bn()
        coco_eval.evaluate_coco(dataset_val,retinanet,method,pr)
    elif(method=="onnx"):
        
        if(pr==32) :
            
            ort_session = onnxruntime.InferenceSession('/kaggle/working/500_int_8_mm_w8a8.onnx')
        elif(pr==16) : 
            
            ort_session = onnxruntime.InferenceSession('/kaggle/working/500_int_8_mm_w8a8.onnx')
        coco_eval.evaluate_coco(dataset_val,ort_session,method,pr)
    elif(method=="tvm"): 
        onnx_model_path = '/kaggle/input/int_8_onnx_model/onnx/retina_net/1/400_int_8_mm_w8a8.onnx'
        onnx_model = onnx.load(onnx_model_path)
        input_shape = (1,3,640,640)
        input_name = "input.1"
        shape_dict ={input_name:input_shape}
        mod,params = relay.frontend.from_onnx(onnx_model,shape_dict)
        target = "llvm"
        with tvm.transform.PassContext(opt_level=3):
            executor = relay.build_module.create_executor("graph",mod,tvm.cpu(0),target,params).evaluate()
        coco_eval.evaluate_coco(dataset_val,executor,method,pr)
 
