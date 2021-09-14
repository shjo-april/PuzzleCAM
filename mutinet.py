import os
import sys
import copy
import shutil
import random
import numpy as np



import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

from torch.utils.data import DataLoader

from core.networks import *
from core.datasets import *
from core.mutinet_utils import *



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
from omegaconf import OmegaConf, dictconfig
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

imagenet_mean = [0.485, 0.456, 0.406]
imagenet_std = [0.229, 0.224, 0.225]



class MNTrainer:

    def __init__(self, cfg: dictconfig.DictConfig) -> None:
        self.CONFIG = cfg
        set_seed(0)


        # region LOG & VISUALIZATION

        data_dir = create_directory(f'./experiments/data/')
        self.data_dic = {
            'train': [],
            'validation': [],
        }
        self.train_timer = Timer()
        self.eval_timer = Timer()

        log_dir = create_directory(f'./experiments/logs/')
        log_path = log_dir + f'{ self.CONFIG.tag}.txt'
        self.data_path = data_dir + f'{ self.CONFIG.tag}.json'
        self.log_func = lambda string='': log_print(string, log_path)

        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())
        tensorboard_dir = create_directory(
            f'./experiments/tensorboards/{ self.CONFIG.tag}/{TIMESTAMP}/')
        self.writer = SummaryWriter(tensorboard_dir)

        self.log_func('[i] {}'.format(self.CONFIG.tag))
        self.log_func()

        # endregion

        # region DATATSET TRANSFORM
        domainA = 'train_1_2_f'
        domainB = 'train_1_2_b'
        domainA = [image_id.strip() for image_id in open(
            './data/%s.txt' % domainA).readlines()]
        domainB = [image_id.strip() for image_id in open(
            './data/%s.txt' % domainB).readlines()]
        self.domainset: DomainSet = DomainSet(domainA, domainB)

        self.PseMaker = TestPseMaker(cfg)

        self.new_pse_dir = './experiments/predictions/{}/'.format(
            cfg.tag)

        train_transforms = [
            RandomResize_For_Segmentation(
                self.CONFIG.min_image_size,  self.CONFIG.max_image_size),
            RandomHorizontalFlip_For_Segmentation(),

            Normalize_For_Segmentation(imagenet_mean, imagenet_std),
            RandomCrop_For_Segmentation(self.CONFIG.image_size),
        ]

        # if 'Seg' in args.architecture:
        #     if 'C' in args.architecture:
        #         train_transforms.append(Resize_For_Mask(args.image_size // 4))
        #     else:
        #         train_transforms.append(Resize_For_Mask(args.image_size // 8))

        self.train_transform = transforms.Compose(
            train_transforms + [Transpose_For_Segmentation()])

        self.test_transform = transforms.Compose([
            Normalize_For_Segmentation(imagenet_mean, imagenet_std),
            Top_Left_Crop_For_Segmentation(self.CONFIG.image_size),
            Transpose_For_Segmentation()
        ])

        meta_dic = read_json('./data/VOC_2012.json')
        class_names = np.asarray(meta_dic['class_names'])

        self.log_func('[i] train_transform is {}'.format(self.train_transform))
        self.log_func()
        # endregion

        # region NETWORK


        the_number_of_gpu =  torch.cuda.device_count()
        if the_number_of_gpu > 1:
            self.log_func(
                '[i] the number of gpu : {}'.format(the_number_of_gpu))

        self.model_dir = create_directory(
            f'./experiments/models/{self.CONFIG.tag}/')
        self.save_model_fn = lambda model, step: save_model(
            model, self.model_dir + f'best_{step}.pth', parallel=the_number_of_gpu > 1)

        self.ModelA = None
        self.ModelB = None

        for  name in  ['A', 'B']:
            model = None
            if self.CONFIG.architecture == 'DeepLabv3+':
                model = DeepLabv3_Plus(
                    self.CONFIG.backbone, num_classes=meta_dic['classes'] + 1, mode=self.CONFIG.mode, use_group_norm=self.CONFIG.use_gn)
            elif self.CONFIG.architecture == 'Seg_Model':
                model = Seg_Model(self.CONFIG.backbone,
                                  num_classes=meta_dic['classes'] + 1)
            elif self.CONFIG.architecture == 'CSeg_Model':
                model = CSeg_Model(self.CONFIG.backbone,
                                   num_classes=meta_dic['classes'] + 1)
           
            model = model.cuda()
            model.train()

            if the_number_of_gpu > 1:
                model = nn.DataParallel(model)
            load_model(
                model, self.CONFIG['init_model'+name], parallel=the_number_of_gpu > 1)

            if(name == 'A'):
                self.ModelA = model
            elif(name == 'B'):  
                self.ModelB = model

        self.log_func('[i] Architecture is {}'.format(
            self.CONFIG.architecture))
        self.log_func('[i] Total Params: %.2fM' %
                      (calculate_parameters(model)))

        # for sync bn
        # patch_replication_callback(model)

        # endregion

    def __del__(self):
        self.writer.close()

    def _evaluate(self, model, iteration):
        valid_dataset = VOC_Dataset_For_Segmentation(
            self.CONFIG.data_dir, 'val', self.test_transform)
        loader = DataLoader(valid_dataset, batch_size=self.CONFIG.batch_size,
                            num_workers=1, shuffle=False, drop_last=True)
        model.eval()
        self.eval_timer.tik()
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
                    for b in range(self.CONFIG.batch_size):
                        image = get_numpy_from_tensor(images[b])
                        pred_mask = get_numpy_from_tensor(predictions[b])

                        image = denormalize(
                            image, imagenet_mean, imagenet_std)[..., ::-1]
                        h, w, c = image.shape

                        pred_mask = decode_from_colormap(
                            pred_mask, valid_dataset.colors)
                        pred_mask = cv2.resize(
                            pred_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                        image = cv2.addWeighted(
                            image, 0.5, pred_mask, 0.5, 0)[..., ::-1]
                        image = image.astype(np.float32) / 255.

                        self.writer.add_image(
                            'Mask/{}'.format(b + 1), image, iteration, dataformats='HWC')

                for batch_index in range(images.size()[0]):
                    pred_mask = get_numpy_from_tensor(predictions[batch_index])
                    gt_mask = get_numpy_from_tensor(labels[batch_index])

                    h, w = pred_mask.shape
                    gt_mask = cv2.resize(
                        gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

                    meter.add(pred_mask, gt_mask)

                sys.stdout.write(
                    '\r# Evaluation [{}/{}] = {:.2f}%'.format(step + 1, length, (step + 1) / length * 100))
                sys.stdout.flush()

        print(' ')
        model.train()
        return meter.get(clear=True)

    def train(self, model, domain, label_path, step):
        class_loss_fn = nn.CrossEntropyLoss(ignore_index=255).cuda()

        train_dataset = VOC_Dataset_For_WSSS(
            self.CONFIG.data_dir, domain, label_path, self.train_transform)
        train_loader = DataLoader(train_dataset, batch_size=self.CONFIG.batch_size,
                                  num_workers=self.CONFIG.num_workers, shuffle=True, drop_last=True)

        val_iteration = len(train_loader)
        log_iteration = int(val_iteration * self.CONFIG.print_ratio)
        max_iteration = self.CONFIG.max_epoch * val_iteration

        self.log_func('[i] log_iteration : {:,}'.format(log_iteration))
        self.log_func('[i] val_iteration : {:,}'.format(val_iteration))
        self.log_func('[i] max_iteration : {:,}'.format(max_iteration))

        train_meter = Average_Meter(['loss'])

        best_valid_mIoU = -1
        param_groups = model.module.get_parameter_groups(None)
        params = [
            {'params': param_groups[0], 'lr': self.CONFIG.lr,
                'weight_decay': self.CONFIG.wd},
            {'params': param_groups[1], 'lr': 2 *
                self.CONFIG.lr, 'weight_decay': 0},
            {'params': param_groups[2], 'lr': 10 *
                self.CONFIG.lr, 'weight_decay': self.CONFIG.wd},
            {'params': param_groups[3], 'lr': 20 *
                self.CONFIG.lr, 'weight_decay': 0},
        ]

        # self.log_func('[i] The number of pretrained weights : {}'.format(len(param_groups[0])))
        # self.log_func('[i] The number of pretrained bias : {}'.format(len(param_groups[1])))
        # self.log_func('[i] The number of scratched weights : {}'.format(len(param_groups[2])))
        # self.log_func('[i] The number of scratched bias : {}'.format(len(param_groups[3])))

        optimizer = PolyOptimizer(params, lr=self.CONFIG.lr, momentum=0.9,
                                  weight_decay=self.CONFIG.wd, max_step=max_iteration, nesterov=self.CONFIG.nesterov)

        train_iterator = Iterator(train_loader)

        torch.autograd.set_detect_anomaly(True)

        for iteration in range(max_iteration):
            images, labels = train_iterator.get()
            images, labels = images.cuda(), labels.cuda()

            #################################################################################################
            # Inference
            #################################################################################################
            logits = model(images)

            ###############################################################################
            # The part is to calculate losses.
            ###############################################################################
            if 'Seg' in self.CONFIG.architecture:
                labels = resize_for_tensors(labels.type(torch.FloatTensor).unsqueeze(
                    1), logits.size()[2:], 'nearest', None)[:, 0, :, :]
                labels = labels.type(torch.LongTensor).cuda()

                # print(labels.size(), labels.min(), labels.max())

            loss = class_loss_fn(logits, labels)
            #################################################################################################

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_meter.add({
                'loss': loss.item(),
            })

            #################################################################################################
            # For Log
            #################################################################################################
            if (iteration + 1) % log_iteration == 0:
                loss = train_meter.get(clear=True)
                learning_rate = float(
                    get_learning_rate_from_optimizer(optimizer))

                data = {
                    'iteration': iteration + 1,
                    'learning_rate': learning_rate,
                    'loss': loss,
                    'time': self.train_timer.tok(clear=True),
                }
                self.data_dic['train'].append(data)
                write_json(self.data_path, self.data_dic)

                self.log_func('[i] \
                    iteration={iteration:,}, \
                    learning_rate={learning_rate:.4f}, \
                    loss={loss:.4f}, \
                    time={time:.0f}sec'.format(**data)
                              )

                self.writer.add_scalar('Train/loss', loss, iteration)
                self.writer.add_scalar(
                    'Train/learning_rate', learning_rate, iteration)

            #################################################################################################
            # Evaluation
            #################################################################################################
            if (iteration + 1) % val_iteration == 0:
                mIoU, _ = self._evaluate(model, iteration)

                if best_valid_mIoU == -1 or best_valid_mIoU < mIoU:
                    best_valid_mIoU = mIoU
                    self.save_model_fn(model, step+"%.1f"%best_valid_mIoU)
                    self.log_func('[i] save model')

                data = {
                    'iteration': iteration + 1,
                    'mIoU': mIoU,
                    'best_valid_mIoU': best_valid_mIoU,
                    'time': self.eval_timer.tok(clear=True),
                }
                self.data_dic['validation'].append(data)
                write_json(self.data_path, self.data_dic)

                self.log_func('[i] \
                    iteration={iteration:,}, \
                    mIoU={mIoU:.2f}%, \
                    best_valid_mIoU={best_valid_mIoU:.2f}%, \
                    time={time:.0f}sec'.format(**data)
                              )

                self.writer.add_scalar('Evaluation/mIoU', mIoU, iteration)
                self.writer.add_scalar(
                    'Evaluation/best_valid_mIoU', best_valid_mIoU, iteration)

            write_json(self.data_path, self.data_dic)

    def _make_pseduo(self, step, save_path):
        self.domainset=self.PseMaker(self.ModelA, self.ModelB,
                      self.domainset, step, save_path)

    def run(self):

        new_pse_path = create_directory(self.new_pse_dir+'/newpse/')
        # new_pse_path = self.CONFIG.pselabel_path
        # self.train(self.ModelA, self.domainset.dominFor_A,
        #            new_pse_path, 'test')
        # return
        for i in range(1):

            # step = f'A_{i}'
            # new_pse_path =  create_directory(self.new_pse_dir+f'/{step}/')
            # self._make_pseduo(step,new_pse_path)
            # # break
            # self.train(self.ModelA, self.domainset.dominFor_A,
            #            new_pse_path, step)

            step = f'B_{i}'
            new_pse_path =  create_directory(self.new_pse_dir+f'/{step}/')
            self._make_pseduo(step,new_pse_path)
            self.train(self.ModelB, self.domainset.dominFor_B,
                       new_pse_path, step)
        pass


if __name__ == '__main__':
    config_path = 'config/voc12.yaml'
    CONFIG = OmegaConf.load(config_path)

    trainer = MNTrainer(CONFIG)
    trainer.run()
    print(type(CONFIG))
    del trainer
