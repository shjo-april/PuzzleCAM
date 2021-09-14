import matplotlib.pyplot as plt

import sys
import numpy as np
import torch

sys.path.insert(0, '..')

device = torch.device('cuda:0')


from hRnet import init_predictor
from hRnet.baseclicker import Clicker

MODEL_THRESH=0.49


# Possible choices: 'NoBRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C', 'RGB-BRS', 'DistMap-BRS'
predictor = init_predictor()


def get_iou(gt_mask, pred_mask, ignore_label=-1):
    ignore_gt_mask_inv = gt_mask != ignore_label
    obj_gt_mask = gt_mask == 1

    intersection = np.logical_and(np.logical_and(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()
    union = np.logical_and(np.logical_or(pred_mask, obj_gt_mask), ignore_gt_mask_inv).sum()

    return intersection / union

class Hr_predictor:

    def __init__(self) -> None:
        self.TARGET_IOU = 0.49
        self.max_iou_thr=0.95
        self.min_clicks=1  
        self.clicker=Clicker() 

    def hrnet_predict(self,sample,max_points=1,callback=None):
        max_clicks=max_points
        gt_mask = sample['pse_mask']
        image   = sample['image']

        self.clicker = Clicker(gt_mask=gt_mask)
        pred_mask = np.zeros_like(gt_mask)
        ious_list = []

        with torch.no_grad():
            predictor.set_input_image(image)

            for click_indx in range(max_clicks):
                self.clicker.make_next_click(pred_mask)
                pred_probs = predictor.get_prediction(self.clicker)
                pred_mask = pred_probs > MODEL_THRESH

                if callback is not None:
                    callback(image, gt_mask, pred_probs, click_indx, self.clicker.clicks_list)

                iou = get_iou(gt_mask, pred_mask)
                ious_list.append(iou)

                if iou >=self.max_iou_thr and click_indx + 1 >= self.min_clicks:
                    break

            # return clicker.clicks_list, np.array(ious_list, dtype=np.float32), pred_probs


        pred_mask = pred_probs > MODEL_THRESH

