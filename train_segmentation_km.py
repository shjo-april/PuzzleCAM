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
from  spml.utils.segsort.common import kmeans,generate_location_features,kmeans_with_initial_labels,initialize_cluster_labels

import spml.utils.general.common as common_utils

from train_util import  get_spixel_image

TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,0,6"

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
# Hyperparameter
###############################################################################
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--max_epoch', default=50, type=int)

parser.add_argument('--lr', default=0.007, type=float)
parser.add_argument('--wd', default=4e-5, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=256, type=int)
parser.add_argument('--max_image_size', default=512, type=int)

parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='train_kmeans66_SALImages', type=str)

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
    
    meta_dic = read_json('./data/VOC_2012.json')
    class_names = np.asarray(meta_dic['class_names'])


    
    # train_dataset = VOC_Dataset_For_WSSS(args.data_dir, 'train_aug', 'VOC2012/VOCdevkit/VOC2012/saliency_map/', train_transform)
    train_dataset = VOC_Dataset_For_MNSS(
        args.data_dir, 'VOC2012/VOCdevkit/VOC2012/SALImages/' ,'train_aug',train_transform)
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
    
    ###################################################################################
    # Network
    ###################################################################################
    if args.architecture == 'DeepLabv3+':
        model = DeepLabv3_Plus(args.backbone, num_classes=meta_dic['classes'] + 1, mode=args.mode, use_group_norm=args.use_gn)
    elif args.architecture == 'Seg_Model':
        model = Seg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    elif args.architecture == 'CSeg_Model':
        model = CSeg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)

    param_groups = model.get_parameter_groups(None)
    params = [
        {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
        {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
        {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
        {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    ]
    
    model = model.cuda()
    model.train()
    # model.load_state_dict(torch.load('experiments/models/train_kmeans66.pth'))

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
        logits = model(images)
        _, _, h, w = logits.size()
        tagpred = F.avg_pool2d(logits, kernel_size=(h, w), padding=0)#tagpred[7]
        loss_cls = F.multilabel_soft_margin_loss(tagpred[:, 1:].view(tagpred.size(0), -1), labels[:,1:])
        preds=F.softmax(logits,1)# preds[0,:,0,0]
        N, C ,H,W= images.shape

        if(False):
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

            valid_channel = (iou_saliency >= 0.0).view(b, 20, 1, 1).expand(size=(b, 20, h, w))
            
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
       
    


        preds= preds.transpose(1,3).transpose(1,2)
        logits= logits.transpose(1,3).transpose(1,2)

        sailencys= sailencys.transpose(1,3).transpose(1,2)
        # N, H,W,C = feats.shape

        class_loss_list=[]
        for i in range(N):
            with torch.no_grad():
                local_features = generate_location_features(
                    (H, W), 0,feature_type='float')
                local_features -= 0.5
                local_features = local_features.view(1, H, W, 2).expand(args.batch_size, H, W, 2)
                cur_embeddings = images[i].view(-1, C)
                cur_local_features = (
                local_features[i].view(-1, local_features.shape[-1]))
                cur_embeddings_with_loc = torch.cat(
                    [cur_embeddings, cur_local_features], -1)
                cur_embeddings_with_loc = normalize_embedding(
                    cur_embeddings_with_loc)
                cur_embeddings_with_loc=torch.cat([cur_embeddings_with_loc,1.5*sailencys[i].view(-1,1)],dim=1)

                labels22 = initialize_cluster_labels([32,32], [H,W],0)
                res = kmeans_with_initial_labels(  # cur_cluster_indices max:35 min:0  shape: torch.Size([14976]) 聚类算法
                    cur_embeddings_with_loc,
                    labels22.view(-1),#cur_cluster_indices.cpu().numpy()
                    None,
                    16)
                mean_values = torch.tensor( [0.485, 0.456, 0.406]).view(3, 1, 1).cuda()
                spixel_viz, spixel_label_map = get_spixel_image((images[i]/5 + mean_values).clamp(0, 1), res.view(H,W).squeeze(), n_spixels= 256,  b_enforce_connect=True)
                # spixl_save_name = os.path.join('experiments/res/spxiel_viz/',   imgids[i] +'_sPixel.png')
                # cv2.imwrite(spixl_save_name,spixel_viz.transpose(1, 2, 0)*255)
                        
                embeddings_512= torch.cat([preds[i],3*sailencys[i]],dim=2)#
                label_map_512=torch.from_numpy(spixel_label_map).cuda() 
                protos_200_salient=calculate_prototypes_from_labels(embeddings_512,label_map_512)#sailencys[i]..cpu().numpy()[:,21]
                labels_init = initialize_cluster_labels([1,2], [1,protos_200_salient.shape[0]],0)
                label_map_200 = kmeans_with_initial_labels( 
                        protos_200_salient,
                        labels_init.view(-1),#protos_200.detach().cpu().numpy()[0].sum()
                        None,
                        16)
            if(label_map_200.max()<1):
                continue
            new_w=preds[i]
            # new_w= normalize_embedding( # np.unique(protos_200_salient.().cpu().numpy().count(1))
            protos_200=calculate_prototypes_from_labels(new_w,label_map_512,with_norm=False)#label_map_200.detach().cpu().numpy()[:,21]
            protos_sailent=calculate_prototypes_from_labels(sailencys[i],label_map_512,with_norm=False)#label_map_200.detach().cpu().numpy()[:,21]
           
            # protos_bin=calculate_prototypes_from_labels(protos_200,label_map_200,with_norm=False)
            protos_bin_saient=calculate_prototypes_from_labels(protos_200_salient,label_map_200,with_norm=False)
            fg_mask =  labels[i].bool()
            ne_mask=~fg_mask
            fg_mask[0]=False
            bg_mask=labels[i].bool()
            bg_mask=torch.zeros(fg_mask.shape).bool().cuda()
            bg_mask[0]=True
            fg_mask=fg_mask.expand(size=(protos_200.shape[0],21)).bool()
            fg_c=torch.sum(fg_mask*protos_200,dim=1)
            ng_c=torch.sum(ne_mask*protos_200,dim=1)
            bg_c=torch.sum(bg_mask*protos_200,dim=1)#ng_c.detach().cpu().numpy().max()>0.5
            # chanel=torch.stack([fg_c,ng_c,bg_c],dim=0) fg_c+ng_c+bg_c  torch.sum(torch.stack([fg_c,ng_c,bg_c]),dim=0)
            # bg_map = torch.sub(1, bg_c)
            chanel = fg_c
            label33=(protos_sailent>0.3).float()
            # label_map_200=label33.clone()
            # if(protos_bin_saient[0][21]>protos_bin_saient[1][21]) : 
            #     label33[label_map_200==0]=1
            #     label33[label_map_200==1]=0
            # else:
            #     label33[label_map_200==0]=0
            #     label33[label_map_200==1]=1
            label33=label33.squeeze(1).cuda().float()
            loss_c=F.mse_loss(chanel,label33)
            class_loss_list.append(loss_c)
            

            pass
           
        km_loss =torch.mean(torch.stack(class_loss_list))
        ###############################################################################
        # The part is to calculate losses.
        # ###############################################################################
        # if 'Seg' in args.architecture:
        #     labels = resize_for_tensors(labels.type(torch.FloatTensor).unsqueeze(1), logits.size()[2:], 'nearest', None)[:, 0, :, :]
        #     labels = labels.type(torch.LongTensor).cuda()

            # print(labels.size(), labels.min(), labels.max())
        loss= km_loss+loss_cls
        # loss= loss_cls
        # loss = class_loss_fn(bin_logits, bin_mask)

        # loss=torch.tensor(0)
        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss_cls.item(), 
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