# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np
import cv2
import imageio

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.puzzle_utils import *
from core.networks import *
from core.datasets import *

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *

from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *

from hRnet import init_predictor,Clicker,Click

os.environ["CUDA_VISIBLE_DEVICES"] = "5,6,7,8"
parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=4, type=int)
parser.add_argument('--data_dir', default='/media/ders/zhangyumin/DATASETS/dataset/newvoc/VOCdevkit/VOC2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--model_name', default='', type=str)

parser.add_argument('--cam_dir', default='ResNeSt269allPuzzle_schedule@optimal_16@train@scale=0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--domain', default='train_aug', type=str)

parser.add_argument('--beta', default=10, type=int)
parser.add_argument('--exp_times', default=8, type=int)
# parser.add_argument('--threshold', default=0.25, type=float)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    experiment_name = args.model_name

    if 'train' in args.domain:
        experiment_name += '@train'
    else:
        experiment_name += '@val'

    # experiment_name += '@threshold=%.2f'%args.threshold
    experiment_name += '@interas'
    
    cam_dir = f'./experiments/predictions/{args.cam_dir}/'
    pred_dir = create_directory(f'./experiments/predictions/{experiment_name}/')

    model_path = './experiments/models/' + f'{args.model_name}.pth'

    set_seed(args.seed)
    log_func = lambda string='': print(string)

    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    # for mIoU
    meta_dic = read_json('./data/VOC_2012.json')
    dataset = VOC_Dataset_For_Making_CAM(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    path_index = PathIndex(radius=10, default_size=(512 // 4, 512 // 4))

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func()




    
    hr_predictor =init_predictor()
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    pred_thr=0.49
    aa=0
    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, label, gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size
            png_path = pred_dir + image_id + '.png'
            if os.path.isfile(png_path):
                continue
            npy_path = pred_dir + image_id + '.npy'
            if os.path.isfile(npy_path):
                continue

            # preprocessing
            ori_w, ori_h = ori_image.size

            image = np.asarray(ori_image)
            # image = normalize_fn(image)

            # image = torch.from_numpy(image)
            # flipped_image = image.flip(-1)


            # images = torch.stack([image, flipped_image])
            # images = images.cuda()

            # postprocessing
            cam_dict = np.load(cam_dir + image_id + '.npy', allow_pickle=True).item()

            cams = cam_dict['cam'].cpu().numpy()

            cams = np.pad(cams, ((1, 0), (0, 0), (0, 0)), mode='constant', constant_values=0.2)

            cams = np.argmax(cams, axis=0)
            cams = cv2.resize(cams, (0, 0), fx=4, fy=4, interpolation=cv2.INTER_NEAREST)[:ori_h, :ori_w]

            result =np.zeros((ori_h,ori_w))
            for i in range(1,len(cam_dict['keys'])):
                hr_predictor.set_input_image(image)
                gt =  (cams==i)*1.0
                gt[cams==0]=-1
                clicker = Clicker(gt_mask=gt) 
                pred_mask = np.zeros_like(gt)
                clicker.make_next_click(pred_mask) 
                pred_probs = hr_predictor.get_prediction(clicker)
                pred_mask = pred_probs > pred_thr
                result[pred_mask] =cam_dict['keys'][i]
                pass
            ignore_mask= (cams>0)& ((cams>0)!=(result>0))
            result[ignore_mask]=255
            imageio.imwrite(png_path, result.astype(np.uint8))

            # cv2.imwrite("res/succpse/inter_test.png",result*30)
            # cv2.imwrite("res/succpse/cam_test.png",cams*30)
            pass
            print(image_id)

            # unary = unary_from_labels(labels, n_labels, gt_prob=gt_prob, zero_unsure=False)


            
    
    print("python3 evaluate.py --experiment_name {} --domain {}".format(experiment_name, args.domain))