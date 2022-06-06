import cv2
import onnxruntime
import torch
import numpy as np
from torchvision.ops import nms


class HandDetModel(object):
    def __init__(self, capability=0, topk=2, confth=0.3, nmsth=0.45):
        self.capability = capability
        if self.capability == 0:
            model_path = "../lib/models/HandDet.onnx"
        else:
            raise Exception(" [!] Capability == 0 only considered now!")

        self.model = onnxruntime.InferenceSession(model_path)
        self.input_size = self.model.get_inputs()[0].shape[-2:]
        self.topk = topk
        self.confth = confth
        self.nmsth = nmsth
        print("*" * 100)
        print(" [!] HandDet infer-dev:%s model:%s" % (onnxruntime.get_device(), model_path))

    def nms_process(self, num_classes, loc_data, cls_data):
        num_batch = loc_data.shape[0]
        num_priors = loc_data.shape[1]
        loc_data = torch.from_numpy(loc_data)
        conf_data = torch.from_numpy(cls_data)
        output = torch.zeros(num_batch, num_classes, self.topk, 5)  # (8678, 2, 3, 5)
        if num_batch == 1:
            # size batch x num_classes x num_priors
            conf_preds = conf_data.t().contiguous().unsqueeze(0)
        else:
            conf_preds = conf_data.view(num_batch, num_priors, num_classes).transpose(2, 1)
            output.expand_(num_batch, num_classes, self.topk, 5)

        for i in range(num_batch):
            # For each class, perform nms
            decoded_boxes = loc_data[i].clone()
            conf_scores = conf_preds[i].clone()
            for cl in range(1, num_classes):
                c_mask = conf_scores[cl].gt(self.confth)
                scores = conf_scores[cl][c_mask]
                if scores.dim() == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                keep = nms(boxes[:, :4], scores.view(-1), self.nmsth)
                if list(keep.size())[0] > self.topk:
                    keep = keep[: self.topk]
                ids, count = keep, list(keep.size())[0]
                output[i, cl, :count] = torch.cat((scores[ids[:count]].unsqueeze(1), boxes[ids[:count]]), 1)

        return output

    def pre_process(self, img_bgr, is_bgr=True):
        if is_bgr:
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_rgb = img_bgr

        img_resize = cv2.resize(img_rgb, (self.input_size[0], self.input_size[1]))
        img_norm = (img_resize.astype(np.float32) / 255.0 - 0.5) * 2
        img_norm = img_norm.transpose(2, 0, 1)

        return img_norm

    def post_process(self, ort_outs, imw, imh):
        output = self.nms_process(num_classes=2, loc_data=ort_outs[0], cls_data=ort_outs[1])
        dets = output.data
        dets = dets[0, 1, :]
        mask = dets[:, 0].gt(0.0).expand(5, dets.size(0)).t()
        dets = torch.masked_select(dets, mask).view(-1, 5)
        boxes = dets[:, 1:]
        boxes[:, (0, 2)] *= imw
        boxes[:, (1, 3)] *= imh
        scores = dets[:, 0].cpu().numpy()
        cls_dets = np.hstack((boxes.cpu().numpy(), scores[:, np.newaxis])).astype(np.float32, copy=False)
        boxes = cls_dets[:, 0:4]
        hand_boxes = []
        for idx in range(boxes.shape[0]):
            pred_box = [(boxes[idx, 0], boxes[idx, 1]), (boxes[idx, 2], boxes[idx, 3])]
            hand_boxes.append(pred_box)

        return hand_boxes

    def __call__(self, img_rgb):
        imh, imw = img_rgb.shape[:2]
        img_norm = self.pre_process(img_rgb)
        model_outputs = self.model.run(None, {self.model.get_inputs()[0].name: [img_norm]})
        hand_boxes = self.post_process(model_outputs, imw, imh)

        return hand_boxes, None
