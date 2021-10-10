# Copyright (C) 2020 * Ltd. All rights reserved.
# author : Sanghyeon Jo <josanghyeokn@gmail.com>

import os
import sys
import copy
import shutil
import random
import argparse
from cv2 import LMEDS, log
import numpy as np

import torch
from torch import tensor
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader
from imageio import imsave
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
from datetime import datetime

BASE_DIR = r"/media/ders/zhangyumin/SPML"
sys.path.append(BASE_DIR)
sys.path.append(r"/media/ders/zhangyumin/superpixel_fcn")
sys.path.append(r"/media/ders/zhangyumin/PuzzleCAM/")
import core.resnet38d

import models
from loss import *
import train_util

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
parser.add_argument('--architecture', default='Seg_Model', type=str)
parser.add_argument('--backbone', default='resnest50', type=str)
parser.add_argument('--mode', default='fix', type=str)
parser.add_argument('--use_gn', default=True, type=str2bool)

###############################################################################
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--max_epoch', default=20, type=int)

parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--wd', default=4e-5, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='train_kmeansesp_saientcy_resnest50_test', type=str)

parser.add_argument('--label_name', default='AffinityNet@Rresnest269@Puzzle@train@beta=10@exp_times=8@rw@crf=0@color', type=str)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    log_dir = create_directory(f'./experiments/logs/')
    data_dir = create_directory(f'./experiments/data/')
    model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/{TIMESTAMP}/')   
    pred_dir = './experiments/predictions/{}/'.format(args.label_name)
    
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
        RandomResize_For_Segmentation(args.min_image_size, args.max_image_size),
        RandomHorizontalFlip_For_Segmentation(),
        
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        RandomCrop_For_Segmentation(args.image_size),
    ]
    
    # if 'Seg' in args.architecture:
    #     if 'C' in args.architecture:
    #         train_transforms.append(Resize_For_Mask(args.image_size // 4))
    #     else:
    #         train_transforms.append(Resize_For_Mask(args.image_size // 8))

    train_transform = transforms.Compose(train_transforms + [Transpose_For_Segmentation()])
    
    test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(args.image_size),
        Transpose_For_Segmentation()
    ])
    
    meta_dic = read_json('/media/ders/mazhiming/PuzzleCAM-master/data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])


    
    # train_dataset = VOC_Dataset_For_WSSS(args.data_dir, 'train_aug', 'VOC2012/VOCdevkit/VOC2012/saliency_map/', train_transform)
    train_dataset = VOC_Dataset_For_MNSS(
        args.data_dir, '/media/ders/mazhiming/dataset/VOCtrainval_11-May-2012/SALImages/' ,'train_aug',train_transform)
    valid_dataset = VOC_Dataset_For_Segmentation(args.data_dir, 'train', test_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, num_workers=1, shuffle=False, drop_last=True)
    
    log_func('[i] mean values is {}'.format(imagenet_mean))
    log_func('[i] std values is {}'.format(imagenet_std))
    log_func('[i] The number of class is {}'.format(meta_dic['classes']))
    log_func('[i] train_transform is {}'.format(train_transform))
    log_func()

    val_iteration = len(train_loader)
    log_iteration = int(val_iteration * args.print_ratio)
    max_iteration = args.max_epoch * val_iteration
    
    log_func('[i] log_iteration : {:,}'.format(log_iteration))
    log_func('[i] val_iteration : {:,}'.format(val_iteration))
    log_func('[i] max_iteration : {:,}'.format(max_iteration))
    

    network_data = torch.load('/media/ders/zhangyumin/superpixel_fcn/result/VOCAUG/SpixelNet1l_bn_adam_3000000epochs_epochSize6000_b32_lr5e-05_posW0.003_21_09_15_21_42/model_best.tar')
    print("=> using pre-trained model '{}'".format(network_data['arch']))
    #Q_model = models.__dict__[network_data['arch']]( data = network_data).cuda()
    #Q_model = nn.DataParallel(Q_model)
    #Q_model.eval()

    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=meta_dic['classes'] + 1, mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    elif args.architecture == 'resnet38':
        model =  resnet38Net(num_classes=meta_dic['classes'] + 1)
        weights_dict = core.resnet38d.convert_mxnet_to_torch('/media/ders/zhangyumin/PuzzleCAM/experiments/models/train_kmeansesp_saientcy_resnest50.pth')
        model.load_state_dict(weights_dict, strict=False)

    param_groups = model.get_parameter_groups()
    params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    ]
    
    model = model.cuda()
    model.train()
    # model.load_state_dict(torch.load('/media/ders/zhangyumin/PuzzleCAM/experiments/models/train_kmeansesp_saientcy_resnest50_withq_fgmask.pth'))

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

        # for sync bn
        # patch_replication_callback(model)

    load_model_fn = lambda: load_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn = lambda: save_model(model, model_path, parallel=the_number_of_gpu > 1)
    save_model_fn_for_backup = lambda: save_model(model, model_path.replace('.pth', f'_backup.pth'), parallel=the_number_of_gpu > 1)
    
    ###################################################################################
    # Loss, Optimizer
    ###################################################################################
    class_loss_fn = nn.CrossEntropyLoss(ignore_index=255).cuda()

    # log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
    # log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
    # log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
    # log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))
    
    optimizer = PolyOptimizer(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : [],
    }

    train_timer = Timer()
    eval_timer = Timer()

    train_meter = Average_Meter(['loss','km_loss'])

    best_valid_mIoU = -1

    def evaluate(loader):
        model.eval()
        eval_timer.tik()

        meter = Calculator_For_mIoU('./data/VOC_2012.json') 

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels) in enumerate(loader):
                images = images.cuda()
                labels = labels.cuda()
 
                logits = model(images)
                logits=F.softmax(logits,1)
                # logits[:,0,:,:]=0.2
                # output = Q_model(images)
                # logits=train_util.upfeat(logits, output, 16, 16).cuda()
                predictions = torch.argmax(logits, dim=1)
                
                # for visualization
                if step == 0:
                    for b in range(args.batch_size):
                        image = get_numpy_from_tensor(images[b])
                        pred_mask = get_numpy_from_tensor(predictions[b])

                        image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
                        h, w, c = image.shape

                        pred_mask = decode_from_colormap(pred_mask, train_dataset.colors)
                        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        image = cv2.addWeighted(image, 0.5, pred_mask, 0.5, 0)[..., ::-1]
                        image = image.astype(np.float32) / 255.

                        writer.add_image('Mask/{}'.format(b + 1), image, iteration, dataformats='HWC')
                
                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])

                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    meter.add(pred_mask, gt_mask)

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        
        print(' ')
        model.train()
        
        return meter.get(clear=True)
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    torch.autograd.set_detect_anomaly(True)

    for iteration in range(max_iteration):
        images, imgids,labels,masks,sailencys= train_iterator.get()
        images = images.cuda()
        labels = labels.cuda()
        sailencys = sailencys.cuda().view(sailencys.shape[0],1,sailencys.shape[1],sailencys.shape[2])/255.0
        #################################################################################################
        # Inference
        #################################################################################################
        #output = Q_model(images)

        #prob = output.clone().cuda(1)

        logits = model(images)
        _, _, h, w = logits.size()
        sailencys = F.interpolate(sailencys, size=(h, w))
        #sailencys = train_util.poolfeat(sailencys.cuda(1), prob, 16, 16).cuda(0)

        tagpred = F.avg_pool2d(logits, kernel_size=(h, w), padding=0)#
        loss_cls = F.multilabel_soft_margin_loss(tagpred[:, 1:].view(tagpred.size(0), -1), labels[:,1:])
        N, C ,H,W= images.shape
        if(True):
            cam=logits
            b, c, h, w = cam.size()
            sailencys = F.interpolate(sailencys.float(), size=(h, w))

            label_map = labels[:,1:].view(b, 20, 1, 1).expand(size=(b, 20, h, w)).bool()#label_map_bg[0,:,0,0]
            # Map selection
            label_map_fg = torch.zeros(size=(b, 21, h, w)).bool().cuda()
            label_map_bg = torch.zeros(size=(b, 21, h, w)).bool().cuda()

            label_map_bg[:, 0] = True
            label_map_fg[:,1:] = label_map.clone()

            sal_pred = F.softmax(cam, dim=1) 

            iou_saliency = (torch.round(sal_pred[:, 1:].detach()) * torch.round(sailencys)).view(b, 20, -1).sum(-1) / \
                        (torch.round(sal_pred[:, 1:].detach()) + 1e-04).view(b, 20, -1).sum(-1)

            valid_channel = (iou_saliency > 0.4).view(b, 20, 1, 1).expand(size=(b, 20, h, w))
            
            label_fg_valid = label_map & valid_channel

            label_map_fg[:, 1:] = label_fg_valid
            label_map_bg[:, 1:] = label_map & (~valid_channel)

            # Saliency loss
            fg_map = torch.zeros_like(sal_pred).cuda()
            bg_map = torch.zeros_like(sal_pred).cuda()

            fg_map[label_map_fg] = sal_pred[label_map_fg]
            bg_map[label_map_bg] = sal_pred[label_map_bg]

            fg_map = torch.sum(fg_map, dim=1, keepdim=True)
            bg_map = torch.sum(bg_map, dim=1, keepdim=True)
    
            bg_map = torch.sub(1, bg_map) #label_map_fg[1,:,0,0] torch.sum(fg_map[7][0]>0.5) F.mse_loss(2*fg_map,sailencys) 
            sal_pred = fg_map * 0.5 + bg_map * (1 - 0.5) 

            km_loss =F.mse_loss(sal_pred,sailencys)
       
    
        if(False):
            reconstr_feat=(logits).cuda(1)
            for i in range(2):
                reconstr_feat = train_util.upfeat(reconstr_feat, prob, 16, 16)  #reconstr_feat.cpu().numpy()
                reconstr_feat = train_util.poolfeat(reconstr_feat, prob, 16, 16)
            q_loss =F.mse_loss(logits,reconstr_feat.cuda(0))


        ###############################################################################
        # The part is to calculate losses.
        # ###############################################################################
        # if 'Seg' in args.architecture:
        #     labels = resize_for_tensors(labels.type(torch.FloatTensor).unsqueeze(1), logits.size()[2:], 'nearest', None)[:, 0, :, :]
        #     labels = labels.type(torch.LongTensor).cuda()

            # print(labels.size(), labels.min(), labels.max())
        alpha=1
        if(iteration<2*log_iteration):
            alpha=0.3
        if(iteration<5*log_iteration):
            alpha=0.6
        #loss= km_loss+loss_cls+q_loss*alpha
        loss= loss_cls + km_loss
        # loss = class_loss_fn(bin_logits, bin_mask)

        # loss=torch.tensor(0)
        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 
            'km_loss' : km_loss.item(), 
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss,km_loss = train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'km_loss' : km_loss, 
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                km_loss={km_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % val_iteration == 0:
            mIoU, _ = evaluate(valid_loader)
            
            if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                best_valid_mIoU = mIoU

                save_model_fn()
                log_func('[i] save model')

            data = {
                'iteration' : iteration + 1,
                'mIoU' : mIoU,
                'best_valid_mIoU' : best_valid_mIoU,
                'time' : eval_timer.tok(clear=True),
            }
            data_dic['validation'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                mIoU={mIoU:.2f}%, \
                best_valid_mIoU={best_valid_mIoU:.2f}%, \
                time={time:.0f}sec'.format(**data)
            )
            
            writer.add_scalar('Evaluation/mIoU', mIoU, iteration)
            writer.add_scalar('Evaluation/best_valid_mIoU', best_valid_mIoU, iteration)
    
    write_json(data_path, data_dic)
    writer.close()

    print(args.tag)