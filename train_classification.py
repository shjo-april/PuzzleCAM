# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

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

parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=8, type=int)
parser.add_argument('--data_dir', default='../VOCtrainval_11-May-2012/', type=str)

###############################################################################
# Network
###############################################################################
parser.add_argument('--architecture', default='resnet50', type=str)
parser.add_argument('--mode', default='normal', type=str) # fix

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=15, type=int)

parser.add_argument('--lr', default=0.1, type=float)
parser.add_argument('--wd', default=1e-4, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='', type=str)
parser.add_argument('--augment', default='', type=str)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    log_dir = create_directory(f'./experiments/logs/')
    data_dir = create_directory(f'./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/')
    
    log_path = log_dir + f'{args.tag}.txt'
    data_path = data_dir + f'{args.tag}.json'
    model_path = model_dir + f'{args.tag}.pth'

    set_seed(args.seed)
    log_func = lambda string='': log_print(string, log_path)
    
    log_func('[i] {}'.format(args.tag))
    log_func()
    
    ###################################################################################
    # Transform, Dataset, DataLoader
    ###################################################################################
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    normalize_fn = Normalize(imagenet_mean, imagenet_std)
    
    train_transforms = [
        RandomResize(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip(),
    ]
    if 'colorjitter' in args.augment:
        train_transforms.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1))
    
    if 'randaugment' in args.augment:
        train_transforms.append(RandAugmentMC(n=2, m=10))
    
    train_transform = transforms.Compose(train_transforms + \
        [
            Normalize(imagenet_mean, imagenet_std),
            RandomCrop(args.image_size),
            Transpose()
        ]
    )
    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])
    
    train_dataset = VOC_Dataset_For_Classification(args.data_dir, 'train_aug', train_transform)

    train_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'train', test_transform)
    valid_dataset_for_seg = VOC_Dataset_For_Testing_CAM(args.data_dir, 'val', test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    train_loader_for_seg = DataLoader(train_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)
    valid_loader_for_seg = DataLoader(valid_dataset_for_seg, batch_size=args.batch_size, num_workers=1, drop_last=True)

    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func('[i] test_transform is {}'.format(test_transform))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration

    # val_iteration = log_iteration

    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    
    ###################################################################################
    # Network
    ###################################################################################
    model = Classifier(args.architecture, meta_dic['classes'], mode=args.mode)
    param_groups = model.get_parameter_groups(print_fn=None)
    
    model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu = '0'

    the_number_of_gpu = len(use_gpu.split(','))
    if the_number_of_gpu > 1:
        log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
        model = nn.DataParallel(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.MultiLabelSoftMarginLoss(reduction='none').cuda()

    log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))
    
    optimizer = PolyOptimizer([
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    ], lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : []
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss', 'class_loss'])
    
    best_train_mIoU = -1
    thresholds = list(np.arange(0.10, 0.50, 0.05))

    def evaluate(loader):
        model.eval()
        eval_timer.tik()

        meter_dic = {th : Calculator_For_mIoU('./data/VOC_2012.json') for th in thresholds}

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels, gt_masks) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()

                _, features = model(images, with_cam=True)
                
                # features = resize_for_tensors(features, images.size()[-2:])
                # gt_masks = resize_for_tensors(gt_masks, features.size()[-2:], mode='nearest')

                mask = labels.unsqueeze(2).unsqueeze(3)
                cams = (make_cam(features) * mask)

                # for visualization
                if step == 0:
                    obj_cams = cams.max(dim=1)[0]
                    
                    for b in range(8):
                        image = get_numpy_from_tensor(images[b])
                        cam = get_numpy_from_tensor(obj_cams[b])

                        image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
                        h, w, c = image.shape

                        cam = (cam * 255).astype(np.uint8)
                        cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_LINEAR)
                        cam = colormap(cam)

                        image = cv2.addWeighted(image, 0.5, cam, 0.5, 0)[..., ::-1]
                        image = image.astype(np.float32) / 255.

                        writer.add_image('CAM/{}'.format(b + 1), image, iteration, dataformats='HWC')

                for batch_index in range(images.size()[0]):
                    # c, h, w -> h, w, c
                    cam = get_numpy_from_tensor(cams[batch_index]).transpose((1, 2, 0))
                    gt_mask = get_numpy_from_tensor(gt_masks[batch_index])

                    h, w, c = cam.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    for th in thresholds:
                        bg = np.ones_like(cam[:, :, 0]) * th
                        pred_mask = np.argmax(np.concatenate([bg[..., np.newaxis], cam], axis=-1), axis=-1)

                        meter_dic[th].add(pred_mask, gt_mask)

                # break

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        
        print(' ')
        model.train()
        
        best_th = 0.0
        best_mIoU = 0.0

        for th in thresholds:
            mIoU, mIoU_foreground = meter_dic[th].get(clear=True)
            if best_mIoU < mIoU:
                best_th = th
                best_mIoU = mIoU

        return best_th, best_mIoU
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)                                                          

    for iteration in range(max_iteration):
        images, labels = train_iterator.get()
        images, labels = images.cuda(), labels.cuda()

        #################################################################################################
        logits = model(images)

        class_loss = class_loss_fn(logits, labels).mean()
        loss = class_loss
        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 
            'class_loss' : class_loss.item()
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss, class_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'class_loss' : class_loss,
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)

            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                class_loss={class_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/class_loss', class_loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            threshold, mIoU = evaluate(train_loader_for_seg)
            
            if best_train_mIoU == -1 or best_train_mIoU < mIoU:
                best_train_mIoU = mIoU

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'threshold' : threshold,
                'train_mIoU' : mIoU,
                'best_train_mIoU' : best_train_mIoU,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                threshold={threshold:.2f}, \
                train_mIoU={train_mIoU:.2f}%, \
                best_train_mIoU={best_train_mIoU:.2f}%, \
                time={time:.0f}sec'.format(**data)
            )
            
            writer.add_scalar('Evaluation/threshold', threshold, iteration)
            writer.add_scalar('Evaluation/train_mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_train_mIoU', best_train_mIoU, iteration)
    
    write_json(data_path, data_dic)
    writer.close()

    print(args.tag)