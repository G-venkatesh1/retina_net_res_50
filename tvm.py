import torch
import tvm
import numpy as np
# from tvm import relay
from torchvision import transforms
import onnx
import time
from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer,Resizer_const
from retinanet import coco_eval
from tqdm import tqdm
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
onnx_model_path = '/kaggle/input/int_8_onnx_model/onnx/retina_net/1/400_int_8_mm_w8a8.onnx'
onnx_model = onnx.load(onnx_model_path)
input_shape = (1,3,640,640)
input_name = "input.1"
shape_dict ={input_name:input_shape}
mod,params = relay.frontend.from_onnx(onnx_model,shape_dict)
target = "llvm"
with tvm.transform.PassContext(opt_level=3):
    executor = relay.build_module.create_executor("graph",mod,tvm.cpu(0),target,params).evaluate()
    
dataset_val = CocoDataset('/kaggle/input/coco-2017-dataset/coco2017', set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer_const()]))
x = dataset_val[0]['img'].numpy()
input_data = tvm.nd.array(x.astype("float32")) 
output = executor(input_data).numpy()
print(output.shape)
    
