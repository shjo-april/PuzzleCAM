import os
import sys
import copy
import shutil
import random
import numpy as np
from numpy.core.fromnumeric import sort
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *
from core.mutinet_utils import *


from tools.ai.log_utils import *
from tools.general.io_utils import *
from tools.general.time_utils import *
from tools.general.json_utils import *
from tools.general.txt_utils import *

from tools.ai.torch_utils import *
from tools.ai.evaluate_utils import *


from tools.ai.augment_utils import *
from tools.ai.randaugment import *
import numpy as np
class DomainSet:
    def __init__(self,dominA,dominB,dominCom=[]) -> None:
        self.dominA=dominA
        self.dominB=dominB
        self.dominCom=dominCom

    @property
    def dominFor_A(self):
        return self.dominA +self.dominCom

    @property
    def dominFor_B(self):
        return self.dominB +self.dominCom

    
class BasePseMaker:
    def __init__(self,cfg) -> None:
        self.CONFIG=cfg
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        self.test_transform = transforms.Compose([
        Normalize_For_Segmentation(imagenet_mean, imagenet_std),
        Top_Left_Crop_For_Segmentation(self.CONFIG.image_size),
        Transpose_For_Segmentation()
        ])
        palette_img_PIL = Image.open(r"/media/ders/zhangyumin/irn-master/VOCdevkit/VOC2012/SegmentationClass/2007_000039.png")
        self.palette = palette_img_PIL.getpalette()

class TestPseMaker(BasePseMaker):
    def __init__(self,cfg) -> None:
        super().__init__(cfg)
    
    def __call__(self,ModelA,ModelB,dm:DomainSet,cur_step,savepath) -> DomainSet:
        l_model,r_model = (ModelA,ModelB) if cur_step[0]=='B' else (ModelB,ModelA)
        l_model.eval(),r_model.eval()

        valid_dataset = VOC_Dataset_For_MNSS(
        self.CONFIG.data_dir, self.CONFIG.pselabel_path,'train_aug', self.test_transform)
        loader = DataLoader(valid_dataset, batch_size=self.CONFIG.batch_size,
                            num_workers=1, shuffle=False, drop_last=False)
        # model.eval()
        meter1 = Calculator_For_mIoU('./data/VOC_2012.json')
        meter2 = Calculator_For_mIoU('./data/VOC_2012.json')
        ignore_meter = Average_Meter(['ignore_rate'])
        ignore_zip_list=[]
        with torch.no_grad():
            length = len(loader)
            for step, (images, imgids,sizes,labels,pse_labels) in enumerate(loader):

                images = images.cuda()
                labels = labels.cuda()
                pse_labels = pse_labels.cuda()
                l_logits = l_model(images)
                r_logits = r_model(images)

                # predictions = torch.argmax(l_logits, dim=1)
                pass
                new_pse_labels,ignore_rate = self._make_pse(l_logits,r_logits,pse_labels)
                ignore_meter.add({
                    'ignore_rate' : torch.mean(ignore_rate).item(), 
                })
                ignore_zip_list += zip(list(imgids),ignore_rate.cpu().numpy().tolist())

                for batch_index in range(images.size()[0]):
                    w,h=sizes[0][batch_index].item(),sizes[1][batch_index].item()
                    gt_mask = get_numpy_from_tensor(labels[batch_index])[0:h,0:w]


                    new_pse_label =get_numpy_from_tensor(new_pse_labels[batch_index])[0:h,0:w]
                    pred_mask = new_pse_label
                  
                    ignore_gt=np.array(gt_mask, copy=True)
                    ignore_gt[pred_mask!=255]=0
                    meter2.add(ignore_gt,gt_mask)
                    gt_mask[pred_mask==255]=255 
                    meter1.add(pred_mask, gt_mask)
                    #region save
                    img_path=os.path.join(savepath,imgids[batch_index]+'.png')
                    img_pil= Image.fromarray(pred_mask.astype(np.uint8))
                    img_pil.putpalette(self.palette)
                    img_pil.save(img_path)
                    #endregion

                sys.stdout.write(
                    '\r# MakePse [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()
        
        print(' ')
        l_model.train(),r_model.train()
        print('meter1: ')
        print(meter1.get(detail=True,clear=True))
        print('meter2: ')
        print(meter2.get(detail=True,clear=True))
        print("avg_ignore_rate",ignore_meter.get(clear=True))

        dm_ori= dm.dominFor_A if cur_step[0]=='A' else  dm.dominFor_B
        res =[x for x in ignore_zip_list  if not x[0] in dm_ori ]
        res.sort(key=lambda x:x[1])
        dm.dominCom+=list(zip(*res))[0][0:int(len(res)*0.2)]
        write_txt(os.path.join(savepath,'tmp_domain.txt'),  dm.dominCom)
        dfb= dm.dominFor_B
        dfb=sorted(set(dfb),key=dfb.index)

        return dm
    def _make_pse(self,l_logits,r_logits,origin_pses):
        l_preds = torch.argmax(l_logits, dim=1)
        r_preds = torch.argmax(r_logits, dim=1)
        mask_l2opse=l_preds != origin_pses
        mask_l2r = l_preds != r_preds
        # mask_l2r = True
        # ignore_mask = mask_l2opse|mask_l2r #sum(mask_l2r[origin_pses!=255].sum())
        ignore_mask = mask_l2opse
        l_preds[ignore_mask]=255
        ignore_rate= (ignore_mask&(origin_pses!=255)).sum(dim=[1,2])/(origin_pses.shape[1]*origin_pses.shape[2]-(origin_pses==255).sum(dim=[1,2]))
        return l_preds,ignore_rate

