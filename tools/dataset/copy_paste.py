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
import tqdm
from torchvision.transforms import Compose, CenterCrop, ToTensor, Resize
# from tools.ai.augment_utils import Normalize
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
def save_colored_mask(mask, save_path):
    lbl_pil = Image.fromarray(mask.astype(np.uint8), mode="P")
    colormap = imgviz.label_colormap()
    lbl_pil.putpalette(colormap.flatten())
    lbl_pil.save(save_path)


def random_flip_horizontal(mask, img, p=0.5):
    if np.random.random() < p:
        img = img[:, ::-1, :] #（H,W,3） img.numpy()[:, ::-1, :]
        mask = mask[:, ::-1]  #（H,W）
    return mask, img


def img_add(img_src, img_main, mask_src):
    if len(img_main.shape) == 3:
        h, w, c = img_main.shape
    elif len(img_main.shape) == 2:
        h, w = img_main.shape
    mask = np.asarray(mask_src.cpu(), dtype=np.uint8)
    img_ret = img_main.clone()
    img_ret2= img_main.clone()
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


def Large_Scale_Jittering(mask, img, min_scale=0.8, max_scale=0.9):
    rescale_ratio = np.random.uniform(min_scale, max_scale)   #0.2<alph<1
    h, w,_= img.shape
    # rescale
    h_new, w_new = int(h * rescale_ratio), int(w * rescale_ratio)

  

    # normalize_fn = Normalize(imagenet_mean, imagenet_std)

    img.resize_(size=(w_new,h_new,3))
    mask.resize_(size=(w_new,h_new))

   
    # mask = mask.resize((w_new, h_new), Image.NEAREST)

    # crop or padding
    x, y = int(np.random.uniform(0, abs(w_new - w))), int(np.random.uniform(0, abs(h_new - h)))
    if rescale_ratio <= 1.0:  # padding
        img_pad = torch.ones((h, w,3), dtype=img.dtype) * 168   #grag
      
        img_pad[..., 0] = (img_pad[..., 0] / 255. - mean[0]) / std[0]
        img_pad[..., 1] = (img_pad[..., 1] / 255. - mean[1]) / std[1]
        img_pad[..., 2] = (img_pad[..., 2] / 255. - mean[2]) / std[2]
        mask_pad = torch.zeros((h, w), dtype=mask.dtype)  
        img_pad[y:y+h_new, x:x+w_new, :] = img
        mask_pad[y:y+h_new, x:x+w_new] = mask
        return mask_pad, img_pad
    else:  # crop
        img_crop = img[y:y+h, x:x+w, :]
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


def data_augment(images,sailencys):
    
    image=images.clone()
    sailency=sailencys.clone()
    batch_size=sailencys.shape[0]  
    for i in range(batch_size):
        chose=np.random.choice(batch_size)#   
       
        if np.random.random() < 0.5: 
            sailencys[i], images[i] = Large_Scale_Jittering(sailencys[i], images[i])
            image[i] = img_add(images[chose], images[i], sailencys[chose])
            nsailency=sailency[chose] 
            nsailency[nsailency>0]=5 
            sailency[i] = img_add(nsailency, sailency[i], sailencys[chose])
       
        #         writer.add_image('disp/{}'.format(args.batch_size*step+b + 1), displacement, iteration, dataformats='HWC')
        # mask_filename =str(imgids[i])+'.png'
        # img_filename = mask_filename.replace('.png', '.jpg')
        # save_colored_mask(sailency[i], os.path.join('experiments/demo/',mask_filename))
        # cv2.imwrite(os.path.join('experiments/demo/', img_filename), image[i])
        # sailency[i+2*batch_size],image[i+2*batch_size]=copy_paste(sailencys[chose2],images[chose2],sailencys[i],images[i])
   
    
    return image,sailency


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






