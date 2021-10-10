# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import imageio
from numpy.core.fromnumeric import take

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
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
parser.add_argument('--architecture', default='DeepLabv3+', type=str)
parser.add_argument('--backbone', default='resnest50', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--use_gn', default=True, type=str2bool)

###############################################################################
# Inference parameters
###############################################################################
parser.add_argument('--tag', default='train_kmeans66', type=str)

parser.add_argument('--domain', default='train', type=str)

parser.add_argument('--scales', default='0.5,1.0,1.5,2.0', type=str)
parser.add_argument('--iteration', default=0, type=int)
parser.add_argument('--thr', default=0.4, type=float)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()

    model_dir = create_directory('./experiments/models/')
    model_path = model_dir + f'{args.tag}.pth'

    if 'train' in args.domain:
        args.tag += '@train'
    else:
        args.tag += '@' + args.domain
    
    args.tag += '@scale=%s'%args.scales
    args.tag += '@iteration=%d'%args.iteration
    args.tag += '@thr=%f'%args.thr

    pred_dir = create_directory('./experiments/predictions/{}/'.format(args.tag))
    
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
    dataset = VOC_Dataset_For_Evaluation(args.data_dir, args.domain)
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=meta_dic['classes'] + 1, mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    
    model = model.cuda()
    model.eval()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    load_model(model, model_path, parallel=False)
    
    #################################################################################################
    # Evaluation
    #################################################################################################
    eval_timer = Timer()
    scales = [float(scale) for scale in args.scales.split(',')]
    
    model.eval()
    eval_timer.tik()

    def inference(images, image_size):
        images = images.cuda()
        
        logits = model(images)
        logits = resize_for_tensors(logits, image_size)#torch.max(logits[0][1:])
        logits = F.softmax(logits, dim=1)
        logits = logits[0] + logits[1].flip(-1) #logits[:,0,0]
        logits = get_numpy_from_tensor(logits).transpose((1, 2, 0))
        return logits 

    with torch.no_grad():
        length = len(dataset)
        for step, (ori_image, image_id, tag,gt_mask) in enumerate(dataset):
            ori_w, ori_h = ori_image.size
            cams_list = []

            for scale in scales:
                image = copy.deepcopy(ori_image)
                image = image.resize((round(ori_w*scale), round(ori_h*scale)), resample=PIL.Image.CUBIC)
                
                image = normalize_fn(image)
                image = image.transpose((2, 0, 1))

                image = torch.from_numpy(image)
                flipped_image = image.flip(-1)
                
                images = torch.stack([image, flipped_image])

                cams = inference(images, (ori_h, ori_w))
    

                cams_list.append(cams)
            
            preds = np.sum(cams_list, axis=0)
            
            if args.iteration > 0:
                # h, w, c -> c, h, w
                preds = crf_inference(np.asarray(ori_image), preds.transpose((2, 0, 1)), t=args.iteration)
                pred_mask = np.argmax(preds, axis=0)
            else:
                # print(np.max(preds, (0, 1), keepdims=True) )

                norm_cam = preds / (np.max(preds, (0, 1), keepdims=True) + 1e-5)
                for i in range(21):
                    if(tag[i]==0):
                        norm_cam[:,:,i]=0
                    else:
                        pass
                        # print(norm_cam[:,:,i])

                norm_cam[:,:,0]=args.thr
                # print(norm_cam)
                # print(norm_cam[100][100][0])

                # print(preds[0][0])
                # print(preds[0])
                pred_mask = np.argmax(norm_cam, axis=-1)


            ###############################################################################
            # cv2.imwrite('./demo.jpg', np.concatenate([np.asarray(ori_image)[..., ::-1], decode_from_colormap(pred_mask, dataset.colors)], axis=1))
            # input('write')

            # cv2.imshow('Image', np.asarray(ori_image)[..., ::-1])
            # cv2.imshow('Prediction', decode_from_colormap(pred_mask, dataset.colors))
            # cv2.imshow('GT', decode_from_colormap(gt_mask, dataset.colors))
            # cv2.waitKey(0)
            ###############################################################################

            if args.domain == 'test':
                pred_mask = decode_from_colormap(pred_mask, dataset.colors)[..., ::-1]
            
            imageio.imwrite(pred_dir + image_id + '.png', pred_mask.astype(np.uint8))
            
            sys.stdout.write('\r# Make CAM [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
            sys.stdout.flush()
        print()
    
    if args.domain == 'val':
        print("python3 evaluate.py --experiment_name {} --domain {} --mode png".format(args.tag, args.domain))