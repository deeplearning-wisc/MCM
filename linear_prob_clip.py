import argparse
import math
import os
import torch
import numpy as np
from tqdm import tqdm
import logging
import clip
from torch.utils.data import DataLoader
# from torchvision import transforms
from sklearn.utils import shuffle
from models.linear import LinearClassifier
import torch.backends.cudnn as cudnn
from utils import *

def set_up_logger(args):
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(args.log_directory, "linear_probe_info.log"), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 
    return log

def save_model_clf(args, classifier, optimizer, epoch, save_file):
    print('==> Saving...')
    state = {
        'opt': args,
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    del state


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def set_optimizer(args, model):
    optimizer = torch.optim.SGD(model.parameters(),
                          lr=args.learning_rate,
                          momentum=args.momentum,
                          weight_decay=args.weight_decay)
    return optimizer

def set_model(args):
    if args.model == 'clip':
        featurizer, preprocess = clip.load(args.ckpt, args.device)
    classifier = LinearClassifier(feat_dim=args.feat_dim, num_classes=args.n_cls).cuda()
    return preprocess, featurizer, classifier

def linear_probe_one_epoch(args, train_loader, featurizer, classifier, criterion, optimizer, epoch, log):
    """one epoch training"""
    featurizer.eval()
    classifier.train()
    losses = AverageMeter()
    top1 = AverageMeter()

    for idx, (images, labels) in enumerate(train_loader):
        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]
        # warm-up learning rate
        warmup_learning_rate(args, epoch, idx, len(train_loader), optimizer)

        # compute loss
        with torch.no_grad():
            features = featurizer.encode_image(images).float() #convert from fp16 to fp32
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
        output = classifier(features.detach())
        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1 = accuracy(output, labels, topk=(1,))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # print info
        if (idx + 1) % args.print_freq == 0:
            log.debug('Train: [{0}][{1}/{2}]\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), loss=losses, top1=top1))
    return losses.avg, top1.avg

def validate(args, val_loader, featurizer, classifier, criterion, log):
    """validation"""
    featurizer.eval()
    classifier.eval()
    losses = AverageMeter()
    top1 = AverageMeter()
    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            bsz = labels.shape[0]
            # forward
            features = featurizer.encode_image(images).float()
            if args.normalize: 
                features /= features.norm(dim=-1, keepdim=True)
            output = classifier(features)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1 = accuracy(output, labels, topk=(1, ))
            top1.update(acc1[0], bsz)

            if idx % args.print_freq == 0:
                log.debug('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader),
                       loss=losses, top1=top1))

    log.debug(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg


def parse_option():
    parser = argparse.ArgumentParser('argument for playing with CLIP')
    #dataset 
    parser.add_argument('--in_dataset', type=str, default='ImageNet',
                        choices=['CIFAR-10', 'CIFAR-100','ImageNet10','ImageNet100', 'ImageNet'], help='img dataset')
    parser.add_argument('--gpu', default=1, type=int,
                        help='the GPU indice to use')
    #model setup
    parser.add_argument('--model', type=str, default='clip',
                        help='model')
    parser.add_argument('--ckpt', type=str, default='ViT-L/14',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='which pretrained img encoder to use')
    parser.add_argument('--feat_dim', type=int, default=768, help='feat dim')
    parser.add_argument('--normalize', action='store_true',
                        help='whether the feautures are normalized')
    #optimization basic
    parser.add_argument('--epochs', type=int, default=40,
                        help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='init lr')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--batch_size', type=int, default=512,
                        help='batch_size')
    # if linear lr decay (default)
    parser.add_argument('--lr_decay_epochs', type=str, default='20,30,35',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    #if cosine lr decay
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    #if warm up lr (default true)
    parser.add_argument('--warm', action='store_false',
                        help='warm-up for large batch training')
    #logging & saving
    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency (# of batch)')
    parser.add_argument('--save_freq', type=int, default=10,
                        help='save frequency (# of epoch)')
    parser.add_argument('--unique_id', type=str, default='test_place_holder',
                        help='id of the run')
    parser.add_argument("--server", type=str, default='inst-01', help="run on which server")
    args = parser.parse_args()

    args.device = f"cuda:{args.gpu}"
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True

    args.lr_decay_epochs = [int(epoch) for epoch in args.lr_decay_epochs.split(",")]
    CKPT_MARKER = {'ViT-B/32':'ViT-B-32', 'ViT-B/16':'ViT-B-16', 'ViT-L/14':'ViT-L-14'}
    args.unique_id = '{}_{}_lr_{}_decay_{}_bsz_{}_{}'.\
        format(args.in_dataset, CKPT_MARKER[args.ckpt], args.learning_rate, args.weight_decay,
               args.batch_size, args.unique_id)
    if args.cosine:
        args.unique_id = '{}_cosine'.format(args.unique_id)
    # warm-up for large-batch training,
    if args.warm:
        args.unique_id = '{}_warm'.format(args.unique_id)
        args.warmup_from = 0.01
        args.warm_epochs = 10
        if args.cosine:
            eta_min = args.learning_rate * (args.lr_decay_rate ** 3)
            args.warmup_to = eta_min + (args.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * args.warm_epochs / args.epochs)) / 2
        else:
            args.warmup_to = args.learning_rate
    if args.server in ['inst-01', 'inst-04']:
        args.save_dir = f'/nobackup/checkpoints/clip_linear/{args.in_dataset}'
        args.root_dir = '/nobackup/dataset_myf'
    if args.server in ['galaxy-01']:
        args.save_dir = f'/nobackup/checkpoints/clip_linear/{args.in_dataset}'
        args.root_dir = '/nobackup-slow/dataset'
    args.log_directory = "linear_probe_logs/{in_dataset}/{unique_id}/".format(in_dataset=args.in_dataset, unique_id= args.unique_id)
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_directory, exist_ok=True)

    return args

def linear_probe_pytorch():
    args = parse_option()
    
    # set up training 
    if args.in_dataset in ['CIFAR-10', 'ImageNet10']:
        args.n_cls = 10
    elif args.in_dataset in ['CIFAR-100', 'ImageNet100']:
        args.n_cls = 100
    elif args.in_dataset == "ImageNet":
        args.n_cls = 1000

    log = set_up_logger(args)
    preprocess, featurizer, classifier = set_model(args)
    val_loader = set_val_loader(args, preprocess)
    train_loader = set_train_loader(args, preprocess)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = set_optimizer(args, classifier)

    # training routine
    best_acc = 0
    for epoch in range(1, args.epochs + 1):
        adjust_learning_rate(args, optimizer, epoch)
        # train for one epoch
        loss, acc = linear_probe_one_epoch(args, train_loader, featurizer, classifier, criterion,
                          optimizer, epoch, log)
        log.debug('Train epoch {}, loss: {:2f}, accuracy:{:.2f}'.format(
            epoch, loss, acc))
        # eval for one epoch
        loss, val_acc = validate(args, val_loader, featurizer, classifier, criterion, log)
        if val_acc > best_acc:
            best_acc = val_acc
        if epoch % args.save_freq == 0:
            save_file = os.path.join(
                args.save_dir, f'{args.unique_id}_linear_probe_epoch_{epoch}.pth')
            save_model_clf(args, classifier, optimizer, epoch, save_file)

def linear_probe_sklearn():
    '''
    train a logistic regression on top of frozen image features extracted by CLIP image encoder (ViT or ResNet)
    CPU version with sklearn
    '''
    args = parse_option()
    # set up training 
    if args.in_dataset in ['CIFAR-10', 'ImageNet10']:
        args.n_cls = 10
    elif args.in_dataset in ['CIFAR-100', 'ImageNet100']:
        args.n_cls = 100
    elif args.in_dataset == "ImageNet":
        args.n_cls = 1000
    preprocess, model, classifier = set_model(args)
    val_loader = set_val_loader(args, preprocess)
    train_loader = set_train_loader(args, preprocess)
    from sklearn.linear_model import LogisticRegression

    train_features, train_labels = get_features(args, model, train_loader, to_np = True)
    test_features, test_labels = get_features(args, model, val_loader, to_np = True)
    # Note: C is the inverse of regularization strength; must be a positive float. 
    # Like in support vector machines, smaller values specify stronger regularization.
    classifier = LogisticRegression(random_state=0, C=0.316, max_iter=1000, verbose=1)
    classifier.fit(train_features, train_labels)
    # Evaluate using the logistic regression classifier
    predictions = classifier.predict(test_features)
    accuracy = np.mean((test_labels == predictions).astype(float)) * 100.
    print(f"Accuracy = {accuracy:.3f}")

if __name__ == '__main__':
    linear_probe_pytorch()
    # linear_probe_sklearn()