
import sys, os
import time
import numpy as np
import torch
import clip
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
import torchvision.transforms as transforms
from continuum.datasets import ImageNet100

from utils.common import AverageMeter, accuracy, warmup_learning_rate

def set_train_loader(args, preprocess = None, batch_size = None, shuffle = False, subset = False):
    # normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
    #                                       std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #for c-10
    # normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) #for c-100
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                        std=(0.26862954, 0.26130258, 0.27577711)) # for CLIP
        preprocess = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if batch_size is None:  #normal case: used for trainign
        batch_size = args.batch_size
        shuffle = True
    if args.in_dataset == "CIFAR-10":
        train_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR10( os.path.join(root, 'cifar10'), train=True, download=True, transform=preprocess),
                    batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "CIFAR-100":
        train_loader = torch.utils.data.DataLoader(
                    datasets.CIFAR100(os.path.join(root, 'cifar100'), train=True, download=True, transform=preprocess),
                    batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "ImageNet":
        if args.server in ['inst-01', 'inst-04']:
            path = os.path.join('/nobackup','ImageNet','train')
        elif args.server in ['galaxy-01', 'galaxy-02']:
            path = os.path.join(root, 'ILSVRC-2012', 'train')
        dataset = datasets.ImageFolder(path, transform=preprocess)
        if subset: 
            from collections import defaultdict
            classwise_count = defaultdict(int)
            indices = []
            for i, label in enumerate(dataset.targets): 
                if classwise_count[label] < args.max_count:
                    indices.append(i)
                    classwise_count[label] += 1
            dataset = torch.utils.data.Subset(dataset, indices)            
        train_loader = torch.utils.data.DataLoader(dataset,
                batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "ImageNet10":
        train_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(os.path.join(root, 'ImageNet10', 'train'), transform=preprocess),
                batch_size=batch_size, shuffle=shuffle, **kwargs)
    elif args.in_dataset == "ImageNet100":
        train_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(os.path.join(root, 'ImageNet100', 'train'), transform=preprocess),
                batch_size=batch_size, shuffle=shuffle, **kwargs)

    return train_loader

def set_val_loader(args, preprocess = None):
    # normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
    #                                       std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    # normalize = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)) #for c-10
    # normalize = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)) #for c-100
    root = args.root_dir
    if preprocess == None:
        normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                        std=(0.26862954, 0.26130258, 0.27577711)) # for CLIP
        preprocess = transforms.Compose([
        transforms.ToTensor(),
        normalize
        ])
    kwargs = {'num_workers': 4, 'pin_memory': True}
    if args.in_dataset == "CIFAR-10":
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(os.path.join(root, 'cifar10'), train=False,download=True, transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
     
    elif args.in_dataset == "CIFAR-100":
        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100(os.path.join(root, 'cifar100'), train=False, download=True,transform=preprocess),
            batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "ImageNet":
        if args.server in ['inst-01', 'inst-04']:
            path = os.path.join('/nobackup','ImageNet','val')
        elif args.server in ['galaxy-01', 'galaxy-02']:
            path = os.path.join(root, 'ILSVRC-2012', 'val')
        val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(path, transform=preprocess),
                batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "ImageNet10":
        val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(os.path.join(root, 'ImageNet10', 'val'), transform=preprocess),
                batch_size=args.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "ImageNet100":
        val_loader = torch.utils.data.DataLoader(
                datasets.ImageFolder(os.path.join(root, 'ImageNet100', 'val'), transform=preprocess),
                batch_size=args.batch_size, shuffle=False, **kwargs)
    return val_loader

def set_model(args):
    
    # create model
    if args.model in ['resnet18', 'resnet34', 'resnet50', 'resnet101']:
        pass
        # model = SupCEHeadResNet(name=args.model, feat_dim = args.feat_dim, num_classes = args.n_cls)


    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))

    torch.cuda.set_device(args.gpus[0]) # sets the default GPU and in order to use multi-GPU
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    if len(args.gpus)> 1:
        model = nn.DataParallel(model.to(args.gpus[0]), args.gpus)
    else:
        model = model.cuda()
    return model


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = model.encoder(images)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1,))
        top1.update(acc1[0][0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]

            # forward
            output = classifier(model.encoder(images))
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1, ))
            top1.update(acc1[0][0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg
