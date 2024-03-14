import torch
import tvm
import numpy as np
from tvm import relay
from torchvision import transforms
from torchvision.ops import nms
import onnx
import time
from retinanet import model
from retinanet.dataloader import CocoDataset, Resizer, Normalizer,Resizer_const
from retinanet import coco_eval
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
import json

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
threshold=0.05
results = []
image_ids = []
c=0
for index in range(len(dataset_val)):
            data = dataset_val[index]
            scale = data['scale']
            if(c>10):break
            c=c+1
            # run network
                # anchors,classificationn = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
            inputs = data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
            inputs=inputs.numpy()
            input_data = tvm.nd.array(inputs.astype("float32")) 
            out = executor(input_data)
            anchors,classificationn=out[0].asnumpy(),out[1].asnumpy()
                # inputs = inputs.half()
                # print(inputs.dtype)
                # ort_inputs['input.1'] = inputs.cpu().numpy()
                # ort_outs = 
                # ort_outs = model.run(None, ort_inputs)
                # anchors,classificationn = ort_outs[0], ort_outs[1]
            classificationn = torch.tensor(classificationn).cuda()
            anchors = torch.tensor(anchors).cuda()
                # print('after prediction',anchors.shape,classificationn.shape)
            finalResult = [[], [], []]
            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])
            for i in range(classificationn.shape[2]):
                scores = torch.squeeze(classificationn[:, :, i]).cuda()
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])
                    # scores[anchors_nms_idx] = scores[anchors_nms_idx].cuda()
                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
                    # print('in eval',finalScores.shape, finalAnchorBoxesIndexes.shape, finalAnchorBoxesCoordinates.shape)
            # finalScores = finalScores.cpu()
            # finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cpu()
            # finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cpu()
            finalAnchorBoxesCoordinates /= scale

            if finalAnchorBoxesCoordinates.shape[0] > 0:
                # change to (x, y, w, h) (MS COCO standard)
                finalAnchorBoxesCoordinates[:, 2] -= finalAnchorBoxesCoordinates[:, 0]
                finalAnchorBoxesCoordinates[:, 3] -= finalAnchorBoxesCoordinates[:, 1]

                # compute predicted labels and scores
                #for box, score, label in zip(boxes[0], scores[0], labels[0]):
                for box_id in range(finalAnchorBoxesCoordinates.shape[0]):
                    score = float(finalScores[box_id])
                    label = int(finalAnchorBoxesIndexes[box_id])
                    box = finalAnchorBoxesCoordinates[box_id, :]

                    # scores are sorted, so we can break
                    if score < threshold:
                        break

                    # append detection for each positively labeled class
                    image_result = {
                        'image_id'    : dataset_val.image_ids[index],
                        'category_id' : dataset_val.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset_val.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset_val)), end='\r')


        # write output
json.dump(results, open('{}_bbox_results.json'.format(dataset_val.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
coco_true = dataset_val.coco
coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset_val.set_name))

        # run COCO evaluation
coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
coco_eval.params.imgIds = image_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()

        # model.train()

  
