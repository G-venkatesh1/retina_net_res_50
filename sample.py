import torch
from torchvision import transforms
import onnxruntime as ort
from retinanet import model,quantise
from retinanet.dataloader import CocoDataset, Resizer, Normalizer,Resizer_const
from retinanet import coco_eval
import onnx
from onnxconverter_common import float16
def main(args=None):
    # model_path ='/kaggle/input/retina_net/pytorch/model/1/coco_resnet_50_map_0_335_state_dict.pt'
    # retinanet = model.resnet50(num_classes=80, pretrained=True).cuda()
    # retinanet.load_state_dict(torch.load(model_path))
    # retinanet.eval()
    # example_input = torch.randn(1, 3,640,640).cuda()
    # onnx_path = '/kaggle/working/fp32_updated.onnx'
    # # torch.onnx.export(retinanet,example_input,onnx_path,opset_version=15)
    # print('export completed')
    #     # model being run
    # ort_session = onnxruntime.InferenceSession('/kaggle/working/fp32_updated.onnx')
    # ort_inputs = {'input.1': None}
    dataset_val = CocoDataset('/kaggle/input/coco-2017-dataset/coco2017', set_name='val2017',
                              transform=transforms.Compose([Normalizer(), Resizer_const()]))
    # co=0
    # for val in range(len(dataset_val)):
    #     d = dataset_val[val]
    #     print(d['img'],d['annot'])
    #     co=co+1
    #     if(co>0):break
    # data = dataset_val[0]
    # inputs = data['img'].permute(2, 0, 1).float().unsqueeze(dim=0)
    # ort_inputs['input.1'] = inputs.cpu().numpy()
    # ort_outs = ort_session.run(None, ort_inputs)
    # anchors,classification = ort_outs[0], ort_outs[1]
    # print(classification.shape,anchors.shape)
    # data = dataset_val[0]
    # inputs = data['img'].permute(2, 0, 1).float().unsqueeze(dim=0)
    # ort_inputs['input.1'] = inputs.cpu().numpy()
    # ort_outs = ort_session.run(None, ort_inputs)
    # anchors,classification = ort_outs[0], ort_outs[1]
    # print(classification.shape,anchors.shape)
    onnx_fp_32_path ='/kaggle/working/retina_net_res_50/fp_32_preprocess.onnx'
    # onnx_fp_16_path = '/kaggle/working/onnx_fp_16.onnx'
    # onnx_32_model = onnx.load(onnx_fp_32_path)
    # onnx_16_model = float16.convert_float_to_float16(onnx_32_model,min_positive_val=1e-7,max_finite_val=1e4)
    # onnx.save(onnx_16_model,onnx_fp_16_path)
    int8_onnx_path ='/kaggle/working/int_8.onnx'
    ort.quantization.shape_inference.quant_pre_process(onnx_fp_32_path, int8_onnx_path)
    module = quantise.OnnxStaticQuantization()
    module.fp32_onnx_path = onnx_fp_32_path
    module.quantization(
        fp32_onnx_path=onnx_fp_32_path,
        future_int8_onnx_path=int8_onnx_path,
        calib_method="Percentile",
        calibration_loader=dataset_val,
        sample=10
    )
    
    
    
if __name__ == '__main__':
    main()
 