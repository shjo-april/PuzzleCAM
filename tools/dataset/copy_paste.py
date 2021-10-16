"""
Unofficial implementation of Copy-Paste for semantic segmentation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import imgviz
import cv2
import argparse
import os
import numpy as np
import random
import tqdm
import  torchvision.transforms as tf
# from tools.ai.augment_utils import Normalize
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
VF = tf.RandomVerticalFlip()
VH = tf.RandomHorizontalFlip()
def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def random_flip_horizontal(mask, image):
    # if np.random.random() < p:
    #     if(len(mask.shape) == 3):  
    #         img = img[:, ::-1, :]  #（H,W,3） img.numpy()[:, ::-1, :]
    #         mask = mask[:, ::-1]   #（H,W）
    #     else:
    #         img = img[:, ::-1, :]  #（H,W,3） img.numpy()[:, ::-1, :]
    #         mask = mask[:, ::-1]   #（H,W）
    if bool(random.getrandbits(1)):
        image = image.flip(dims=[1])
        mask  = mask .flip(dims=[0])
    if bool(random.getrandbits(1)):
        image = image.flip(dims=[2])
        mask  =  mask.flip(dims=[1])
    return mask, image


# def img_add(img_src, img_main, mask_src):
#     if len(img_main.shape) == 3:
#         h, w, c = img_main.shape
#     elif len(img_main.shape) == 2:
#         h, w = img_main.shape
#     mask = np.asarray(mask_src.cpu(), dtype=np.uint8)
#     img_ret = img_main.clone() 
   
#     img_ret[mask]=img_src[mask]     
    # image_show = np.asarray(img_src.cpu()) 
    # image_show2=np.asarray(img_main.cpu())
    # image_show3=np.asarray(img_ret.cpu())

    # norm_image = np.empty_like(image_show, np.int64)
    # norm_image = image_show*25
    # cv2.imwrite('experiments/demo/'+'test_src'+'.png',norm_image)
    # norm_image= image_show2*255   
    # cv2.imwrite('experiments/demo/'+'test_main'+'.png',norm_image)
    # norm_image= image_show3*25    
    # cv2.imwrite('experiments/demo/'+'test_ret'+'.png',norm_image)
    # return img_ret #img_main-img_ret
def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        c,h, w = img_main.shape
        mask = np.asarray(mask_src.cpu(), dtype=np.uint8)
        img_ret = img_main.clone()
        img_ret[0][mask]=img_src[0][mask]  
        img_ret[1][mask]=img_src[1][mask] 
        img_ret[2][mask]=img_src[2][mask]   
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
        mask = np.asarray(mask_src.cpu(), dtype=np.uint8)
        img_ret = img_main.clone()
        img_ret[mask]=img_src[mask]  

    return img_ret #img_main-img_ret
    
def rescale_src(mask_src, img_src, h, w):
    if len(mask_src.shape) == 3:
        h_src, w_src, c = mask_src.shape
    elif len(mask_src.shape) == 2:
        h_src, w_src = mask_src.shape
    max_reshape_ratio = min(h / h_src, w / w_src)
    rescale_ratio = np.random.uniform(0.2, max_reshape_ratio)

    # reshape src img and mask
    rescale_h, rescale_w = int(h_src * rescale_ratio), int(w_src * rescale_ratio)
    mask_src = cv2.resize(mask_src, (rescale_w, rescale_h),
                          interpolation=cv2.INTER_NEAREST)
    # mask_src = mask_src.resize((rescale_w, rescale_h), Image.NEAREST)
    img_src = cv2.resize(img_src, (rescale_w, rescale_h),
                         interpolation=cv2.INTER_LINEAR)

    # set paste coord
    py = int(np.random.random() * (h - rescale_h))
    px = int(np.random.random() * (w - rescale_w))

    # paste src img and mask to a zeros background
    img_pad = np.zeros((h, w, 3), dtype=np.uint8)
    mask_pad = np.zeros((h, w), dtype=np.uint8)
    
    img_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio), :] = img_src
    mask_pad[py:int(py + h_src * rescale_ratio), px:int(px + w_src * rescale_ratio)] = mask_src

    return mask_pad, img_pad


# def Large_Scale_Jittering(mask, img, min_scale=0.2, max_scale=1):
#     rescale_ratio = np.random.uniform(min_scale, max_scale)   #0.2<alph<1
#     h, w,_= img.shape
#     # rescale
#     h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
     
#     # Resize_img  = tf.Resize(size=(w_new, h_new,3))
#     Resize_img   = tf.Resize(size=(w_new, h_new))
#     Resize_mask  = tf.Resize(size=(w_new, h_new),interpolation=0)
#     img  = Resize_img (img.transpose(2,1).transpose(1,0))
#     mask = Resize_mask (mask.unsqueeze(0))
#     img=img.transpose(0,1).transpose(1,2)
#     mask=mask.squeeze(0)
  
   
#     # mask = mask.resize((w_new, h_new), Image.NEAREST)

#     # crop or padding
#     x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
#     if rescale_ratio <= 1.0:  # padding
#         img_pad  = torch.ones ((h, w,3), dtype=img.dtype).cuda() * 168   #grag
#         mask_pad = torch.zeros((h, w  ), dtype=mask.dtype).cuda()        
       
#         img_pad[..., 0] = (img_pad[..., 0] / 255. - mean[1]) / std[1]
#         img_pad[..., 1] = (img_pad[..., 1] / 255. - mean[1]) / std[1]
#         img_pad[..., 2] = (img_pad[..., 2] / 255. - mean[2]) / std[2]

#         img_pad[y:y+h_new, x:x+w_new, :] = img
#         mask_pad[y:y+h_new, x:x+w_new]   = mask   
      
      
#         return mask_pad, img_pad
#     else:  # crop
#         img_crop = img[y:y+h, x:x+w, :]
#         mask_crop = mask[y:y+h, x:x+w]
#         return mask_crop, img_crop
def Large_Scale_Jittering(mask, img, min_scale=0.2, max_scale=1):
    rescale_ratio = np.random.uniform(min_scale, max_scale)   #0.2<alph<1
    _,h, w,= img.shape
    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)
     
    # Resize_img  = tf.Resize(size=(w_new, h_new,3))
    Resize_img   = tf.Resize(size=(w_new, h_new))
    Resize_mask  = tf.Resize(size=(w_new, h_new),interpolation=0)
    img  = Resize_img (img)
    mask = Resize_mask (mask.unsqueeze(0))
    mask=mask.squeeze(0)
  
   
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad  = torch.ones ((3,h, w), dtype=img.dtype).cuda() * 168   #grag
        mask_pad = torch.zeros((h, w  ), dtype=mask.dtype).cuda()        
       
        img_pad[0,...] = (img_pad[0,... ] / 255. - mean[0]) / std[0]
        img_pad[1,...] = (img_pad[1,... ] / 255. - mean[1]) / std[1]
        img_pad[2,...] = (img_pad[2,... ] / 255. - mean[2]) / std[2]

        img_pad[:,y:y+h_new, x:x+w_new] = img
        mask_pad[y:y+h_new, x:x+w_new]   = mask   
      
      
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[:,y:y+h, x:x+w]
        mask_crop = mask[y:y+h, x:x+w]
        return mask_crop, img_crop

def copy_paste(mask_src, img_src, mask_main, img_main,p=0.5,lsj='bool'):
    
    # mask_src, img_src = random_flip_horizontal(mask_src, img_src)
    
    # mask_main, img_main = random_flip_horizontal(mask_main, img_main)
    
        #（9/30）#mask_main, img_main = Large_Scale_Jittering(mask_main, img_main)
        
    #换一下顺序 把scr
    if np.random.random() < p: 
       
       mask_src, img_src = Large_Scale_Jittering(mask_src, img_src)
       mask_main = img_add(mask_src+10, mask_main, mask_src)
       img_main = img_add(img_src, img_main, mask_src)
      
   

    return mask_main, img_main


def data_augment(images,masks):

    # masks,images = random_flip_horizontal(masks, images)
    image=images.clone()
    mask=masks.clone()
    
    batch_size=mask.shape[0]  
    bit=int(np.log2(batch_size))   
    for i in range(batch_size):
        mask[i],image[i]=random_flip_horizontal(mask[i],image[i])
        if np.random.random() < 0.5: 
            chose=random.getrandbits(bit)
            if(chose==i):
                chose=random.getrandbits(bit)
            mask[chose],image[chose]=random_flip_horizontal(mask[chose],image[chose])
            masks_src, images_src = Large_Scale_Jittering(masks[chose], images[chose])
            # masks_src, images_src = VF(masks_src, images_src)
            
            # image_show = np.asarray(images_src.cpu()) 
            # norm_image = np.empty_like(image_show, np.float32)

            # norm_image[..., 0] = (image_show[..., 0]*std[0] +mean[0])*255  
            # norm_image[..., 1] = (image_show[..., 1]*std[1] +mean[1])*255  
            # norm_image[..., 2] = (image_show[..., 2]*std[2] +mean[2])*255 
            # cv2.imwrite('experiments/demo/'+'src_'+str(chose)+'.png',norm_image) mask.max()
            nsailency=masks_src.clone()  
            nsailency[nsailency>0]=10    
            mask_src = np.array(masks_src.cpu(), dtype=np.uint8)
   

            image[i] = img_add(images_src, images[i], masks_src)
            mask [i] = img_add(nsailency ,  masks[i], masks_src)
    
    return image,mask
def new_data_augment(images,masks):

    # masks,images = random_flip_horizontal(masks, images)
    image=images.clone()
    mask=masks.clone()
    
    batch_size=mask.shape[0]  
    bit=int(np.log2(batch_size))   
    for i in range(batch_size): 
        mask[i],image[i]=random_flip_horizontal(mask[i],image[i])
        if np.random.random() < 0.5: 
            chose=random.getrandbits(bit)



    #########get the distance map###########
    
            xyfeat_main,radium_main=get_distance(image[i],mask[i])         #using cv.transpose
            xyfeat_src,radium_src=get_distance(image[chose],mask[chose])   #
    
    #########get the flip、horizontal and resize radio#####################
            flag_flip,flag_horzontal=get_flag(xyfeat_main,xyfeat_src)      #  

            distance=get(xyfeat_main,xyfeat_src)
            radio=(distance-radium_main)/radium_src
            masks_src, images_src = Large_Scale_Jittering(masks[chose], images[chose],radio)
            masks_src, images_src = random_flip_horizontal(flag_flip, flag_horizontal)                          
            # if(chose==i):
            #     chose=random.getrandbits(bit)
            # mask[chose],image[chose]=random_flip_horizontal(mask[chose],image[chose])
            # masks_src, images_src = Large_Scale_Jittering(masks[chose], images[chose])
            # masks_src, images_src = VF(masks_src, images_src)
            
            # image_show = np.asarray(images_src.cpu()) 
            # norm_image = np.empty_like(image_show, np.float32)

            # norm_image[..., 0] = (image_show[..., 0]*std[0] +mean[0])*255  
            # norm_image[..., 1] = (image_show[..., 1]*std[1] +mean[1])*255  
            # norm_image[..., 2] = (image_show[..., 2]*std[2] +mean[2])*255 
            # cv2.imwrite('experiments/demo/'+'src_'+str(chose)+'.png',norm_image) mask.max()
            nsailency=masks_src.clone()  
            nsailency[nsailency>0]=10    
            mask_src = np.array(masks_src.cpu(), dtype=np.uint8)
   

            image[i] = img_add(images_src, images[i], masks_src)
            mask [i] = img_add(nsailency ,  masks[i], masks_src)
    
    return image,mask

def color(mask):
   mask=mask
   return mask

# def src_choose(dir=None):
#     data_dir='/media/ders/zhangyumin/DATASETS/dataset/voc_seg_deeplab/data/VOCtrainval_11-May-2012/VOCdevkit/VOC2012/'
# #    segclass = os.path.join(input_dir, 'SegmentationClass')
# #    JPEGs = os.path.join(input_dir, 'JPEGImages')
# # #    tbar = tqdm.tqdm(masks_path, ncols=100)
# #    for  mask_path in 
#     img_dir=data_dir+'JPEGImages/'
#     gt_dir=data_dir+'/SegmentationClass/'  
    
#     with open('/media/ders/zhangyumin/PuzzleCAM/data/val.txt', 'r') as tf:
#           test_list = tf.readlines()
#     src_path=np.random.choice(test_list) 
#     image=Image.open(img_dir+src_path[:-1]+'.jpg').convert('RGB')
#     mask = np.array(Image.open(gt_dir+src_path[:-1]+'.png'))
#     image = np.asarray(image, dtype=np.float32)
#     mask = np.asarray(mask, dtype=np.int64)
#     return mask,image






