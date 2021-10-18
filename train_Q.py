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
from core.sync_batchnorm.batchnorm import _unsqueeze_ft

from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.Q_util import *


from tools.ai.log_utils import *
from tools.ai.demo_utils import *
from tools.ai.optim_utils import *
from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *

from tools.ai.augment_utils import *
from tools.ai.randaugment import *
from datetime import datetime


import  core.models as fcnmodel
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
start_time=datetime.now().strftime('%Y-%m-%d %H:%M:%S')

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3,0,1"
parser = argparse.ArgumentParser()

###############################################################################
# Dataset
###############################################################################
parser.add_argument('--seed', default=0, type=int)
parser.add_argument('--num_workers', default=2, type=int)
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
parser.add_argument('--batch_size', default=4, type=int)
parser.add_argument('--max_epoch', default=100, type=int)

parser.add_argument('--lr', default=0.0005, type=float)
parser.add_argument('--wd', default=4e-5, type=float)
parser.add_argument('--nesterov', default=True, type=str2bool)

parser.add_argument('--image_size', default=512, type=int)
parser.add_argument('--min_image_size', default=320, type=int)
parser.add_argument('--max_image_size', default=640, type=int)
parser.add_argument('--downsize', default=16, type=int)
parser.add_argument('--print_ratio', default=0.1, type=float)

parser.add_argument('--tag', default='train_Q_relu_new', type=str)

parser.add_argument('--label_name', default='AffinityNet@Rresnest269@Puzzle@train@beta=10@exp_times=8@rw@crf=0@color', type=str)

if __name__ == '__main__':
    ###################################################################################
    # Arguments
    ###################################################################################
    args = parser.parse_args()
    
    # log_dir = create_directory(f'./experiments/logs/')
    # data_dir = create_directory(f'./experiments/data/')
    # model_dir = create_directory('./experiments/models/')
    tensorboard_dir = create_directory(f'./experiments/tensorboards/{args.tag}/{TIMESTAMP}/')   
    pred_dir = './experiments/preditcions/{}/'.format(args.label_name)
    
    log_tag=create_directory(f'./experiments/logs/{args.tag}/')
    data_tag=create_directory(f'./experiments/data/{args.tag}/')
    model_tag=create_directory(f'./experiments/models/{args.tag}/')

    log_path = log_tag+ f'/{start_time}.txt'
    data_path = data_tag + f'/{start_time}.json'
    model_path = model_tag + f'/{start_time}.pth'
    
    
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
    
    # train_dataset = VOC_Dataset_For_WSSS(args.data_dir, 'train_aug', pred_dir, train_transform)
    train_dataset = VOC_Dataset_For_MNSS(
        args.data_dir, 'VOC2012/VOCdevkit/VOC2012/saliency_map/' ,'train_aug',train_transform)
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
    #region 11
    # if args.architecture == 'DeepLabv3+':
    #     model = DeepLabv3_Plus(args.backbone, 9, mode=args.mode, use_group_norm=args.use_gn)
    # elif args.architecture == 'Seg_Model':
    #     model = Seg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)
    # elif args.architecture == 'CSeg_Model':
    #     model = CSeg_Model(args.backbone, num_classes=meta_dic['classes'] + 1)

    # param_groups = model.get_parameter_groups(None)
    # params = [
    #     {'params': param_groups[0], 'lr': args.lr, 'weight_decay': args.wd},
    #     {'params': param_groups[1], 'lr': 2*args.lr, 'weight_decay': 0},
    #     {'params': param_groups[2], 'lr': 10*args.lr, 'weight_decay': args.wd},
    #     {'params': param_groups[3], 'lr': 20*args.lr, 'weight_decay': 0},
    # ]
    #endregion

    model = fcnmodel.SpixelNet1l_bn().cuda()
    model.load_state_dict(torch.load('experiments/models/modelbest17.pth'))


    model = torch.nn.DataParallel(model).cuda()

    #=========== creat optimizer, we use adam by default ==================
    param_groups = [{'params': model.module.bias_parameters(), 'weight_decay': 0},
                    {'params': model.module.weight_parameters(), 'weight_decay': 0}]
    optimizer = torch.optim.Adam(param_groups, args.lr,
                                     betas=(0.9, 0.999))


    # model = model.cuda()
    model.train()

    log_func('[i] Architecture is {}'.format(args.architecture))
    log_func('[i] Total Params: %.2fM'%(calculate_parameters(model)))
    log_func()

    try:
        use_gpu = os.environ['CUDA_VISIBLE_DEVICES']
    except KeyError:
        use_gpu =0
    
    the_number_of_gpu = len(use_gpu.split(','))  
    # if the_number_of_gpu > 1:
    #     log_func('[i] the number of gpu : {}'.format(the_number_of_gpu))
    #     model = nn.DataParallel(model)

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
    
    # optimizer = PolyOptimizer(params, lr=args.lr, momentum=0.9, weight_decay=args.wd, max_step=max_iteration, nesterov=args.nesterov)
    
    #################################################################################################
    # Train
    #################################################################################################
    data_dic = {
        'train' : [],
        'validation' : [],
    }

    train_timer = Timer() #torch.cuda.device_count() 



    eval_timer = Timer()

    train_meter = Average_Meter(['loss','sem_loss','pos_loss','relu_loss'])

    best_valid_mIoU = -1
    spixelID, XY_feat_stack = init_spixel_grid(args)
    val_spixelID,  val_XY_feat_stack = init_spixel_grid(args, b_train=False)
    def evaluate(loader):
        model.eval()
        eval_timer.tik()

        meter = IOUMetric(21) 

        with torch.no_grad():
            length = len(loader)
            for step, (images, labels) in enumerate(loader):  
                images = images.cuda()
                _,_,w,h= images.shape
                labels = labels.cuda()
                inuptfeats=labels.clone()
                inuptfeats[inuptfeats==255]=0 
                inuptfeats=label2one_hot_torch(inuptfeats.unsqueeze(1), C=21)
                inuptfeats=F.interpolate(inuptfeats.float(), size=(12,12),mode='bilinear', align_corners=False)
                inuptfeats=F.interpolate(inuptfeats.float(), size=(w, h),mode='bilinear', align_corners=False)
                with torch.no_grad():
                    prob = model(images)
                inuptfeats=refine_with_q(inuptfeats,prob,20)
                predictions =torch.argmax(inuptfeats,dim=1)
            
                # for visualization
                if step == 0|step == 1:
                    disp=refine_with_q(XY_feat_stack,prob,50) #
                    disp= disp - XY_feat_stack#get_numpy_from_tensor(rgb[0]-rgb[1])
                    b = abs(disp[:,0])-disp[:,0]
                    g = abs(disp[:,0])+disp[:,0]
                    r = abs(disp[:,1])
                    
                    rgb= torch.stack([r,g,b],dim=1) 
                    rgb=make_cam(rgb)
                    mask_fg=(labels>0)
                    mask_fg2=(labels>0)&(labels<21)
                    mask_fg=mask_fg.unsqueeze(1).expand(rgb.shape)
                    rgb[~mask_fg]=1
                    rgb = 200*(1-make_cam(rgb))*mask_fg#rgb[0].cpu().numpy()rgb*mask_fg
                    for b in range(args.batch_size):
                        image = get_numpy_from_tensor(images[b])
                        pred_mask = get_numpy_from_tensor(labels[b])

                        image = denormalize(image, imagenet_mean, imagenet_std)[..., ::-1]
                        h, w, c = image.shape

                        pred_mask = decode_from_colormap(pred_mask, train_dataset.colors)
                        pred_mask = cv2.resize(pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                        
                        image_gt = cv2.addWeighted(image, 0.5, pred_mask, 0.5, 0)[..., ::-1]
                       
                        disp = rgb[b].transpose(0,1).transpose(1,2)
                        displacement =  cv2.addWeighted(image, 0.0, get_numpy_from_tensor(disp).astype(np.uint8), 0.8, 0)[..., ::-1]
                        #displacement
                        image_gt = image_gt.astype(np.float32) / 255.
                        displacement = displacement.astype(np.float32) / 255.

                        writer.add_image('Mask/{}'.format(args.batch_size*step+b + 1), image, iteration, dataformats='HWC')
                        writer.add_image('disp/{}'.format(args.batch_size*step+b + 1), displacement, iteration, dataformats='HWC')
                
                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])

                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                    
                    meter.add_batch(pred_mask, gt_mask)

                sys.stdout.write('\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        
        print(' ')
        model.train()
        
        _,_,_,_,_,_,_,mIoU, _=meter.evaluate()
        return mIoU*100,_
    
    writer = SummaryWriter(tensorboard_dir)
    train_iterator = Iterator(train_loader)

    torch.autograd.set_detect_anomaly(True)
    relufn =nn.ReLU()
    for iteration in range(max_iteration):
        # mIoU, _ = evaluate(valid_loader) 
        images, imgids,tags,masks,sailencys= train_iterator.get()
        tags = tags.cuda()
        sailencys = sailencys.cuda().view(sailencys.shape[0],1,sailencys.shape[1],sailencys.shape[2])/255.0
        labels =(sailencys>0.2).long()#prob[0][4].detach().min()#sailencys[0][0]
        #################################################################################################
        # Inference
        #################################################################################################
        prob = model(images)

        ###############################################################################
        # The part is to calculate losses.
        ###############################################################################
        label_1hot = label2one_hot_torch(labels, C=21) # set C=50 as SSN does
        LABXY_feat_tensor = build_LABXY_feat(label_1hot, XY_feat_stack)  # B* (50+2 )* H * W
            # print(labels.size(), labels.min(), labels.sum())
        loss =torch.tensor(0.0).cuda()
        loss_sem =torch.tensor(0.0).cuda()
        loss_pos =torch.tensor(0.0).cuda()
        relu_loss =torch.tensor(1.0).cuda()
        # prob[:,4]=0.9- relufn(0.9 -prob[:,4]) #prob[:,4].max()
        # relu_loss= 1/(torch.sum(prob,dim=1).mean()**2)
        for gpui in range(the_number_of_gpu):
            curb = args.batch_size/the_number_of_gpu
            # label_1hot_gpui = label_1hot.cuda(gpui)[int(curb*gpui):int(curb*(gpui+1))] # set C=50 as SSN does
            LABXY_feat_tensor_gpui = LABXY_feat_tensor.cuda(gpui)[int(curb*gpui):int(curb*(gpui+1))]  # B* (50+2 )* H * W
            prob_gpui=prob.cuda(gpui)[int(curb*gpui):int(curb*(gpui+1))] 
            loss_guip, loss_sem_guip, loss_pos_guip = compute_semantic_pos_loss( prob_gpui,LABXY_feat_tensor_gpui,
                                                        pos_weight= 0.003, kernel_size=16)
            loss+=loss_guip.cpu()/the_number_of_gpu 
            loss_sem+=loss_sem_guip.cpu()/the_number_of_gpu
            loss_pos+=loss_pos_guip.cpu()/the_number_of_gpu
        # loss, loss_sem, loss_pos = compute_semantic_pos_loss(  prob,LABXY_feat_tensor,
                                                        # pos_weight= 0.003, kernel_size=16)
        # loss+=relu_loss
        # loss = class_loss_fn(logits, labels)
        #################################################################################################
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_meter.add({
            'loss' : loss.item(), 
            'sem_loss' : loss_sem.item(), 
            'pos_loss' : loss_pos.item(), 
            'relu_loss' : relu_loss.item(), 
        })
        
        #################################################################################################
        # For Log
        #################################################################################################
        if (iteration + 1) % log_iteration == 0:
            loss,sem_loss,pos_loss, relu_loss= train_meter.get(clear=True)
            learning_rate = float(get_learning_rate_from_optimizer(optimizer))
            
            data = {
                'iteration' : iteration + 1,
                'learning_rate' : learning_rate,
                'loss' : loss,
                'sem_loss' :sem_loss,
                'pos_loss' :pos_loss,
                'relu_loss' :relu_loss,
                'time' : train_timer.tok(clear=True),
            }
            data_dic['train'].append(data)
            write_json(data_path, data_dic)
            
            log_func('[i] \
                iteration={iteration:,}, \
                learning_rate={learning_rate:.4f}, \
                loss={loss:.4f}, \
                sem_loss={sem_loss:.4f}, \
                pos_loss={pos_loss:.4f}, \
                relu_loss={relu_loss:.4f}, \
                time={time:.0f}sec'.format(**data)
            )

            writer.add_scalar('Train/loss', loss, iteration)
            writer.add_scalar('Train/learning_rate', learning_rate, iteration)
        
        #################################################################################################
        # Evaluation
        #################################################################################################
        if (iteration + 1) % (2*val_iteration) == 0:
            mIoU, _ = evaluate(valid_loader)
            # continue
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