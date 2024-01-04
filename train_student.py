"""
training framework of projector ensemble
"""

from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
import re

from models import model_dict
from models.util import Reg

from dataset.cifar100 import get_cifar100_dataloaders
from dataset.imagenet import get_imagenet_dataloader

from helper.util import adjust_learning_rate

from distiller_zoo import DistillKL

from helper.loops import train_distill as train, validate
from helper.pretrain import init

def parse_option():

    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')
    parser.add_argument('--init_epochs', type=int, default=30, help='init training for two-stage methods')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210', help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--dataset', type=str, default='cifar100', choices=['cifar100','imagenet'], help='dataset')

    # model
    parser.add_argument('--model_s', type=str, default='vgg8',
                        choices=['wrn_16_1', 'wrn_16_2', 'wrn_40_1', 'wrn_40_2',
                                 'vgg8', 'vgg11', 'vgg13', 'vgg16', 'vgg19', 
                                 'resnet8x4', 'MobileNetV2', 'MobileNet', 
                                 'RESNET18','RESNET34','RESNET50'])
    parser.add_argument('--path_t', type=str, default=None, help='teacher model snapshot')

    # distillation
    parser.add_argument('--distill', type=str, default='kd', choices=['kd', 'kd1proj', 'ours'])
    parser.add_argument('--trial', type=str, default='1', help='trial id')

    parser.add_argument('-r', '--gamma', type=float, default=1, help='weight for classification')
    parser.add_argument('-a', '--alpha', type=float, default=None, help='weight balance for KD')
    parser.add_argument('-b', '--beta', type=float, default=None, help='weight balance for other losses')

    # KL distillation
    parser.add_argument('--kd_T', type=float, default=4, help='temperature for KD distillation')

    # hint layer
    parser.add_argument('--gpuNum', default='0', type=str, help='assign gpu number')
    opt = parser.parse_args()

    # set different learning rate 
    if opt.model_s in ['MobileNetV2']:
        opt.learning_rate = 0.01

    # set the path according to the environment
    opt.model_path = './save/student_model'
    opt.tb_path = './save/student_tensorboards'

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_t = get_teacher_name(opt.path_t)

    opt.model_name = 'S:{}_T:{}_{}_{}_r:{}_a:{}_b:{}_{}'.format(opt.model_s, opt.model_t, opt.dataset, opt.distill,
                                                                opt.gamma, opt.alpha, opt.beta, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    return opt


def get_teacher_name(model_path):
    """parse teacher name"""
    segments = model_path.split('/')[-2].split('_')
    if segments[0] != 'wrn':
        return segments[0]
    else:
        return segments[0] + '_' + segments[1] + '_' + segments[2]


def load_teacher(model_path, n_cls):
    print('==> loading teacher model')
    model_t = get_teacher_name(model_path)
    model = model_dict[model_t](num_classes=n_cls)
    
    if model_t in ['RESNET34','RESNET50']:
        model.load_state_dict(torch.load(model_path))
    elif model_t in ['densenet201']:
        pattern = re.compile(r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = torch.load(model_path)
        for key in list(state_dict.keys()):
          res = pattern.match(key)
          if res:
              new_key = res.group(1) + res.group(2)
              state_dict[new_key] = state_dict[key]
              del state_dict[key]
        model = model_dict[model_t](num_classes=n_cls)
        model.load_state_dict(state_dict)
    else:
        model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model
    
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def main():
    opt = parse_option()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpuNum
    
    best_acc = 0

    # fix seed
    print('Seed = ', int(opt.trial)*10)
    setup_seed(int(opt.trial)*10)
    
    # tensorboard logger
    logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

    # dataloader
    if opt.dataset == 'cifar100':
        train_loader, val_loader, n_data = get_cifar100_dataloaders(batch_size=opt.batch_size,
                                                                    num_workers=opt.num_workers,
                                                                    is_instance=True)
        n_cls = 100
    elif opt.dataset == 'imagenet':
        n_cls = 1000
        opt.save_freq = 30
        opt.batch_size = 256
        opt.num_workers = 16
        opt.epochs = 100
        opt.learning_rate = 0.1
        opt.lr_decay_epochs = [30,60,90]
        opt.lr_decay_rate = 0.1
        opt.weight_decay = 1e-4
        opt.momentum = 0.9
        train_loader, val_loader, n_data = get_imagenet_dataloader(batch_size=opt.batch_size, 
                                                                   num_workers=opt.num_workers, 
                                                                   is_instance=True)
    else:
        raise NotImplementedError(opt.dataset)

    # model
    model_t = load_teacher(opt.path_t, n_cls)
    model_s = model_dict[opt.model_s](num_classes=n_cls)

    opt.model_t = get_teacher_name(opt.path_t)    
    print(opt.model_t)
    print(opt.model_s)
    print(opt.distill)

    if opt.dataset == 'cifar100':
        data = torch.randn(2, 3, 32, 32)
    elif opt.dataset == 'imagenet':
        data = torch.randn(2, 3, 224, 224)
        
    model_t.eval()
    model_s.eval()
    feat_t, _ = model_t(data, is_feat=True)
    feat_s, _ = model_s(data, is_feat=True)

    module_list = nn.ModuleList([])
    module_list.append(model_s)
    trainable_list = nn.ModuleList([])
    trainable_list.append(model_s)

    criterion_cls = nn.CrossEntropyLoss()
    criterion_div = DistillKL(opt.kd_T)
    if opt.distill == 'kd':
        criterion_kd = DistillKL(opt.kd_T)
        regress_s = Reg(n_cls, n_cls) 
        module_list.append(regress_s)
    elif opt.distill == 'kd1proj':
        criterion_kd = DistillKL(opt.kd_T)
        regress_s = Reg(n_cls, n_cls) 
        module_list.append(regress_s)
        trainable_list.append(regress_s)
    elif opt.distill == 'ours':        
        criterion_kd = nn.CrossEntropyLoss()     
        # Linear Regress
        _, Cs_h = feat_s[-1].shape              
        _, Ct_h = feat_t[-1].shape
        regress_s1 = Reg(Cs_h, Ct_h) 
        module_list.append(regress_s1)
        regress_s2 = Reg(Cs_h, Ct_h) 
        module_list.append(regress_s2)
        regress_s3 = Reg(Cs_h, Ct_h) 
        module_list.append(regress_s3)
        regress_s4 = Reg(Cs_h, Ct_h) 
        module_list.append(regress_s4)
        trainable_list.append(regress_s1) 
        trainable_list.append(regress_s2) 
        trainable_list.append(regress_s3) 
    else:
        raise NotImplementedError(opt.distill)

    criterion_list = nn.ModuleList([])
    criterion_list.append(criterion_cls)    # classification loss
    criterion_list.append(criterion_div)    # KL divergence loss, original knowledge distillation
    criterion_list.append(criterion_kd)     # other knowledge distillation loss

    # optimizer
    optimizer = optim.SGD(trainable_list.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)

    # append teacher after optimizer to avoid weight_decay
    module_list.append(model_t)

    if torch.cuda.is_available():
        module_list.cuda()
        criterion_list.cuda()
        cudnn.benchmark = True

    # validate teacher accuracy
    teacher_acc, _, _ = validate(val_loader, model_t, criterion_cls, opt)
    print('teacher accuracy: ', teacher_acc)

    # routine
    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        train_acc, train_loss = train(epoch, train_loader, module_list, criterion_list, optimizer, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        logger.log_value('train_acc', train_acc, epoch)
        logger.log_value('train_loss', train_loss, epoch)

        test_acc, tect_acc_top5, test_loss = validate(val_loader, model_s, criterion_cls, opt)

        logger.log_value('test_acc', test_acc, epoch)
        logger.log_value('test_loss', test_loss, epoch)
        logger.log_value('test_acc_top5', tect_acc_top5, epoch)

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'best_acc': best_acc,
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model_s))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model_s.state_dict(),
                'accuracy': test_acc,
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    print(opt.model_t)
    print(opt.model_s)
    print(opt.distill)
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model_s.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model_s))
    torch.save(state, save_file)


if __name__ == '__main__':
    main()
