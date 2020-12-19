import os
import glob
import numpy as np
import argparse

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, Adam

from data.dataset import DogCat
from models.resnet import resnet50


def prepare_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path',
                        default=None,
                        type=str,
                        help='train data path')
    parser.add_argument('--val_data_path',
                        default=None,
                        type=str,
                        help='val data path')
    parser.add_argument('--num_classes',
                        default=2,
                        type=int,
                        help='num clasees of dataset')
    parser.add_argument('--pretrain_path',
                        default=None,
                        type=str,
                        help='Pretrained model path (.pth).')
    parser.add_argument('--image_size',
                        default=256,
                        type=int,
                        help='Height and width of inputs')

    parser.add_argument('--learning_rate',
                        default=0.1,
                        type=float,
                        help='Initial learning rate')
    parser.add_argument('--momentum',
                        default=0.9,
                        type=float,
                        help='Momentum')
    parser.add_argument('--weight_decay',
                        default=1e-3,
                        type=float,
                        help='Weight Decay')

    parser.add_argument('--optimizer',
                        default='sgd',
                        type=str,
                        help='Currently only support SGD')
    parser.add_argument('--lr_scheduler',
                        default='steplr',
                        type=str,
                        help='Type of LR scheduler')

    parser.add_argument('--batch_size',
                        default=32,
                        type=int,
                        help='Batch Size')

    parser.add_argument('--n_epochs',
                        default=100,
                        type=int,
                        help='Number of total epochs to run')

    parser.add_argument('--n_threads',
                        default=4,
                        type=int,
                        help='Number of threads for multi-thread loading')
    parser.add_argument('--checkpoint',
                        default=10,
                        type=int,
                        help='Trained model is saved at every this epochs.')
    parser.add_argument('--manual_seed',
                        default=1,
                        type=int,
                        help='Manually set random seed')

    parser.add_argument('--tensorboard',
                        action='store_true',
                        help='If true, output tensorboard log file.')
    parser.add_argument('--distributed',
                        action='store_true',
                        help='Use multi-processing distributed training to launch N processes per node, which has N GPUs.')
    parser.add_argument('--dist_url',
                        default='tcp://127.0.0.1:23456',
                        type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--world_size',
                        default=-1,
                        type=int,
                        help='number of nodes for distributed training')
    args, _ = parser.parse_known_args()
    return args


def train():
    pass


# def val():
#     paargs

def main():

    args = prepare_parse()
    model = resnet50(num_classes=2)
    optimizer = SGD(model.parameters(),
                    lr=args.learning_rate,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    nesterov=args.nesterov)
    criterion = CrossEntropyLoss()

    train_data = DogCat(args.train_data_path, training=True)
    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
    else:
        train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=(train_sampler is None),
                                               num_workers=args.n_threads,
                                               pin_memory=True,
                                               sampler=train_sampler)

    for ii, (image, label) in enumerate(train_loader):
        image = image.cuda()
        label = label.cuda()
        pred = model(image)
        loss = criterion(pred, label)
        print(loss)




if __name__ == "__main__":
    main()
