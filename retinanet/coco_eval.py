from pycocotools.cocoeval import COCOeval
import json
import torch
from torchvision.ops import nms
import tvm
from tvm import relay
import onnx
def evaluate_coco(dataset, model, threshold=0.05):
    
    # model.eval()
    ort_inputs = {'input.1': None}
    with torch.no_grad():
        c=0
        # start collecting results
        results = []
        image_ids = []
        for index in range(len(dataset)):
            data = dataset[index]
            scale = data['scale']
            if(c>10):break
            c=c+1
            # run network
            if torch.cuda.is_available():
                # anchors,classificationn = model(data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0))
                inputs = data['img'].permute(2, 0, 1).cuda().float().unsqueeze(dim=0)
                # inputs=inputs.numpy()
                # input_data = tvm.nd.array(inputs.astype("float32")) 
                # out = model(input_data)
                # anchors,classificationn=out[0].asnumpy(),out[1].asnumpy()
                # inputs = inputs.half()
                # print(inputs.dtype)
                ort_inputs['input.1'] = inputs.cpu().numpy()
                # ort_outs = 
                ort_outs = model.run(None, ort_inputs)
                anchors,classificationn = ort_outs[0], ort_outs[1]
                classificationn = torch.tensor(classificationn).cuda()
                anchors = torch.tensor(anchors).cuda()
                # print('after prediction',anchors.shape,classificationn.shape)
                finalResult = [[], [], []]
                finalScores = torch.Tensor([])
                finalAnchorBoxesIndexes = torch.Tensor([]).long()
                finalAnchorBoxesCoordinates = torch.Tensor([])
                if torch.cuda.is_available(): 
                    finalScores = finalScores.cuda()
                    finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                    finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()
                for i in range(classificationn.shape[2]):
                    scores = torch.squeeze(classificationn[:, :, i]).cuda()
                    scores_over_thresh = (scores > 0.05)
                    if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                        continue

                    scores = scores[scores_over_thresh].cuda()
                    anchorBoxes = torch.squeeze(anchors).cuda()
                    anchorBoxes = anchorBoxes[scores_over_thresh].cuda()
                    anchors_nms_idx = nms(anchorBoxes, scores, 0.5).cuda()

                    finalResult[0].extend(scores[anchors_nms_idx])
                    finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                    finalResult[2].extend(anchorBoxes[anchors_nms_idx])
                    # scores[anchors_nms_idx] = scores[anchors_nms_idx].cuda()
                    finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                    finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                    if torch.cuda.is_available():
                        finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                    finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                    finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))
                    # print('in eval',finalScores.shape, finalAnchorBoxesIndexes.shape, finalAnchorBoxesCoordinates.shape)
            else:
                print("enable gpu")
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
                        'image_id'    : dataset.image_ids[index],
                        'category_id' : dataset.label_to_coco_label(label),
                        'score'       : float(score),
                        'bbox'        : box.tolist(),
                    }

                    # append detection to results
                    results.append(image_result)

            # append image to list of processed images
            image_ids.append(dataset.image_ids[index])

            # print progress
            print('{}/{}'.format(index, len(dataset)), end='\r')

        if not len(results):
            return

        # write output
        json.dump(results, open('{}_bbox_results.json'.format(dataset.set_name), 'w'), indent=4)

        # load results in COCO evaluation tool
        coco_true = dataset.coco
        coco_pred = coco_true.loadRes('{}_bbox_results.json'.format(dataset.set_name))

        # run COCO evaluation
        coco_eval = COCOeval(coco_true, coco_pred, 'bbox')
        coco_eval.params.imgIds = image_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # model.train()

        return
