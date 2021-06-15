'''
author: CG
data: 2021.3.23
'''

import os
os.environ['KMP_WARNINGS'] = 'off'

import cv2
import argparse
import time
import yaml
import torch
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from utils.profile import count_params
from utils.data_aug import ColorAugmentation_1channel as ColorAugmentation
from torch.autograd.variable import Variable
# sklearn libs
from sklearn.metrics import confusion_matrix
from logger import Logger
import pickle
import roc
import models
# from read_data.read_data_aligned_augment import CASIA as CASIA_AUG
# from read_data.read_data_OULU_CDCN import CASIA
from read_data.read_data import CASIA
from losses import *
from tools.benchmark import compute_speed, stat

# 导入可选模型
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable (models.__dict__[name])
                     )

# 参数设置
parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='models architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('--config', default='cfgs/CDCFeatherNetA-v1.yaml')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=180, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument("--random-seed", type=int, default=14,
                    help='Seed to provide (near-)reproducibility.')
parser.add_argument('--gpus', type=str, default='0', help='use gpus training eg.--gups 0,1')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--lr', '--learning-rate', default=0.01, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='/mnt/cephfs/home/chenguo/code/feather_CDCN/checkpoints/CDCFeatherNetA_nir_032801_0.7_32_0.01000_SGD_train_set_21030601_20210307104514_mask_False/_54_best.pth.tar', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--val', '--evaluate', dest='evaluate', default=True, type=bool,
                    help='evaluate models on validation set')
parser.add_argument('--val-save', default=False, type=bool,
                    help='whether to save evaluate result')
parser.add_argument('--phase-test', default=False, type=bool,
                    help='whether testing in test dataset ')
parser.add_argument('--train_image_list', default='', type=str, help='path to train image list')
parser.add_argument('--input_size', default=112, type=int, help='img crop size')
parser.add_argument('--image_size', default=112, type=int, help='ori img size')
parser.add_argument('--model_name', default='', type=str, help='name of the models')
parser.add_argument('--speed', '--speed-test', default=False, type=bool,
                    help='whether to speed test')
parser.add_argument('--summary', default=False, type=bool,
                    help='whether to analysis network complexity')
parser.add_argument('--every-decay', default=40, type=int, help='how many epoch decay the lr')
parser.add_argument('--fl-gamma', default=3, type=int, help='gamma for Focal Loss')
parser.add_argument('--phase-ir', default=0, type=int, help='phare for IR')
parser.add_argument('--data_flag', type=str, default='20210226', help='use gpus training eg.--gups 0,1')
parser.add_argument('--add_mask',  default=False, type=bool,
                    help='whether to add mask to face image')
best_prec1 = 0

# 导入可选模型
model_names = sorted(name for name in models.__dict__
                     if not name.startswith("__")
                     and callable (models.__dict__[name])
                     )
#检测是否使用GPU
USE_GPU = torch.cuda.is_available()

def test():
    #全局变量
    global args, best_prec1 , USE_GPU , device

    args=parser.parse_args()
    #导入配置文件
    with open(args.config) as f:
        config = yaml.load(f)

    #将config里的属性传给args
    for k, v in config['common'].items():
        setattr(args,k,v)

    #随机种子
    torch.manual_seed(666)
    np.random.seed(666)
    random.seed(666)

    #图片size
    if args.input_size!=256 or args.image_size!=256:
        image_size = args.image_size
        input_size = args.input_size
    else:
        image_size = 256
        input_size = 256
    print("input image size:{},test size{}".format(image_size,input_size))

    #导入模型
    if 'model' in config.keys():
        model = models.__dict__[args.arch](**config['model'])
    else:
        model = models.__dict__[args.arch]()
    #cuda
    device = torch.device('cuda:'+str(args.gpus[0]) if torch.cuda.is_available() else 'cpu')
    #?
    str_input_size = '3x1x256x256'
    # if args.summary:  计算模型复杂度 先不用

    #GPU
    if USE_GPU:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.random_seed)
        args.gpus = [int(i) for i in args.gpus.split(',')]
        model = torch.nn.DataParallel(model,device_ids=args.gpus)#gpu并行计算
        model.to(device)

    #定义loss和optimizer
    criterion = FocalLoss(device,2,gamma=args.fl_gamma)
    optimizer = torch.optim.SGD(model.parameters(),#?
                                args.lr,# 学习率
                                momentum=args.momentum,#加速梯度下降
                                weight_decay=args.weight_decay#正则项的系数，调节复杂度对Loss的影响
                                )
    #if args.speed:  计算耗时，暂不需要
    # 从某处开始恢复
    if args.resume:
        print(os.getcwd())#打印目录
        if os.path.isfile(args.resume):
            print("loading checkpoint from '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            model.module.load_state_dict(checkpoint['model'])
        else:
            print("'{}'is not a checkpoint".format(args.resume))

    #data loading code ??
    #常用的数据预处理，提升泛化能力
    #数据标准化，逐channel的对图像进行标准化，加快模型收敛
    normalize = transforms.Normalize(mean=[0.21735254],std=[0.21561144])

    image_size=args.input_size
    ratio = 112 / float(image_size)#? 112 ?

    #load test data
    print("pack val data")
    #Dataset是一个包装类，把数据包装成Dataset类然后才传入DataLoader中
    val_dataset = CASIA(
        args.data_flag,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        phase_train=False,
        phase_test=args.phase_test
    )

    train_sampler = None
    val_sampler = None
    print("start load val data")
    #dataloader
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        sampler=val_sampler
    )
    print("data load successfully")

    # 评估模型
    if args.evaluate:
        validate(val_loader,[[0.21735254],[1/0.21561144]],
                 model,
                 criterion,
                 args.start_epoch)
        return
    else:
        print(model)#?



#以下为调用函数

#评估模型
def validate(val_loader,param,model,criterion,epoch):
    global time_stp,mask_np
    # 初始化
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    result_list = []
    label_list = []
    predicted_list = []
    # 评估
    model.eval()#进入评估模式，防止改变权值


    end = time.time()

    #测试、计算、打印、记录
    with torch.no_grad():
        #输入的param[0]和[1],其实就是数据标准化的均值和方差
        mean_var = Variable(torch.FloatTensor(param[0]).float().to(device))
        std_var = Variable(torch.FloatTensor(param[1]).float().to(device))
        #dataloder 会返回
        for i,(input,target,depth_dirs) in enumerate(val_loader):
            with torch.no_grad():
                #input val和label
                input_var = Variable(input).float().to(device)
                target_var = Variable(target).long().to(device)
                #输入input，输出判断值
                output = model(input_var)
                loss = criterion (output,target_var)#计算loss
                #测量准确度和记录loss
                prec1,prec2 = accuracy(output.data,target_var,topk=(1,2))#accuracy 函数是自己定义的
                losses.update(loss.data, input.size(0))
                top1.update(prec1[0],input.size(0))

                #通过softmax，判断最大概率，取最大概率对应值为预测值
                #detach是让其不参与参数更新
                soft_output = torch.softmax(output, dim=-1)
                preds = soft_output.to('cpu').detach().numpy()
                label = target.to('cpu').detach().numpy()
                _,predicted = torch.max(soft_output.data,1)#？这行不是很懂
                predicted = predicted.to('cpu').detach().numpy()

                for i_batch in  range(preds.shape[0]):
                    result_list.append(preds[i_batch,1])#?preds第0列是什么？
                    label_list.append(label[i_batch])
                    predicted_list.append(1 if preds[i_batch,1]>= preds[i_batch,0] else 0)#这是通过对比preds的第0与第1列的值来定label

                # if args.val_save: 保存测试结果，暂时用不到

                #记录时间
                batch_time.update(time.time()-end)
                end = time.time()

                if i % args.print_freq == 0: #每过n次就打印一次输出
                    line = 'Test:[{0}/{1}]\t'\
                           'Time {batch_time.val:.3f}({batch_time.avg:.3f})\t'\
                           'Loss {loss.val}:.4f ({loss.avg:.4f})\t'\
                           'Prec@1 {top1.val:.3f}({top1.avg:.3f})\t'\
                            .format(i,len(val_loader),batch_time=batch_time,
                                    loss=losses, top1=top1)
                    #记录该次测试结果  并打印
                    with open('logs/{}_{}.log'.format(time_stp,args.arch),'a+') as flog:
                        flog.write('{}\n'.format(line))
                        print(line)

    confusion_vector = confusion_matrix(label_list,predicted_list)    #由sklearn导入的混淆矩阵
    if confusion_vector.ravel().shape[0]==4:
        tn,fp,fn,tp = confusion_matrix(label_list,predicted_list).ravel() #ravel 拉成一维数组
        #求各评估指标
        fpr = fp/(tn+fp) if (tn+fp)!=0 else 0
        tpr = tp/(tp+fn) if (tp+fn)!=0 else 0
        acer = (fpr+1-tpr)/2
        print ("tn:{},fp:{},fn:{},tp:{}\n,fpr:{},tpr:{},acer:{}\n"
               .format(tn,fp,fn,tp,fpr,tpr,acer))

    else:
        print("test_CG,line 284 wrong,maybe wrong with confusion_matrix,label_list,preds list")

    return top1.avg

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val=0
        self.avg=0
        self.sum=0
        self.count=0

    def update(self,val,n=1):
        self.val=val
        self.sum+=val*n
        self.count+=n
        self.avg = self.sum/self.count

def accuracy(output,target,topk=(1,)):
    #计算准确度、第k个样本的预测值
    maxk = max(topk)
    batch_size = target.size(0)

    _,pred = output.topk(maxk,1,True,True)#求topk 并排序
    pred = pred.t()#?
    correct = pred.eq(target.view(1,-1).expand_as(pred))# view转化为1行，n列，expand到pred的shape

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0,keepdim=True)
        res.append (correct_k.mul_(100.0/batch_size))
    return res

if __name__=='__main__':
    time_stp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())
    test()











