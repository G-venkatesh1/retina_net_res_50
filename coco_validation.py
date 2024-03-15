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


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--model_path', help='Path to model', type=str)

    parser = parser.parse_args(args)

    dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer_const()]))

    # # Create the model
    # retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)

    # use_gpu = True

    # if use_gpu:
    #     if torch.cuda.is_available():
    #         retinanet = retinanet.cuda()

    # if torch.cuda.is_available():
    #     retinanet.load_state_dict(torch.load(parser.model_path))
    #     retinanet = torch.nn.DataParallel(retinanet).cuda()
    # else:
    #     retinanet.load_state_dict(torch.load(parser.model_path))
    #     retinanet = torch.nn.DataParallel(retinanet)

    # retinanet.training = False
    # retinanet.eval()
    # retinanet.module.freeze_bn()
    # ort_session = onnxruntime.InferenceSession('/kaggle/working/400_int_8_mm_w8a8.onnx')
    onnx_model_path = '/kaggle/input/int_8_onnx_model/onnx/retina_net/1/400_int_8_mm_w8a8.onnx'
    onnx_model = onnx.load(onnx_model_path)
    input_shape = (1,3,640,640)
    input_name = "input.1"
    shape_dict ={input_name:input_shape}
    mod,params = relay.frontend.from_onnx(onnx_model,shape_dict)
    target = "llvm"
    with tvm.transform.PassContext(opt_level=3):
        executor = relay.build_module.create_executor("graph",mod,tvm.cpu(0),target,params).evaluate()
    coco_eval.evaluate_coco(dataset_val,executor)
 

if __name__ == '__main__':
    main()
