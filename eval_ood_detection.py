from array import array
import os
import sys
import argparse
import numpy as np
import torch
import clip
from scipy import stats
from models.linear import LinearClassifier
# from torchvision.transforms import transforms
from utils.common import setup_seed, get_num_cls, obtain_cifar_classes, obtain_ImageNet100_classes, obtain_ImageNet10_classes, obtain_ImageNet20_classes, obtain_ImageNet30_classes, obtain_ImageNet_classes
from utils.detection_util import print_measures, get_and_print_results, get_mean_prec, get_Mahalanobis_score, get_ood_scores_clip_linear, get_ood_scores_clip, set_ood_loader, set_ood_loader_ImageNet
from utils.file_ops import prepare_dataframe, save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_model, set_train_loader, set_val_loader
from utils.vit_ops import set_model_clip, set_model_vit, set_model_vit_huggingface, set_train_loader_vit, set_val_loader_vit
# sys.path.append(os.path.dirname(__file__))


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # unique setting for each run
    parser.add_argument('--in_dataset', default='ImageNet10', type=str,
                        choices=['CIFAR-10', 'CIFAR-100',
                                 'ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet30', 'ImageNet100',
                                 'bird200', 'car196', 'flower102', 'food101', 'pet37'], help='in-distribution dataset')
    parser.add_argument('--name', default="test_I10_debug",
                        type=str, help="unique ID for the run")
    # test_imagenet100_10_seed_1
    parser.add_argument('--seed', default=4, type=int, help="random seed")
    parser.add_argument('--gpu', default=[0], nargs='+',
                        help='the GPU indice to use')

    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        help='mini-batch size; 1 for nouns score; 75 for odin_logits; 512 for other scores [clip]')
    # encoder loading
    parser.add_argument('--model', default='CLIP', choices=['CLIP', 'CLIP-Linear', 'H-CLIP',
                        'H-CLIP-Linear', 'vit', 'vit-Linear',  'vit-Linear-H'], type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'RN50x4', 'ViT-L/14'], help='which pretrained img encoder to use')
    # #fine-tune ckpt
    # parser.add_argument('--finetune_ckpt', default =None, type=str,
    #                          help='ckpt location for fine-tuned clip')
    # [linear prob clip] classifier loading
    parser.add_argument('--epoch', default="10", type=str,
                        help='which epoch to test')
    parser.add_argument('--classifier_ckpt', default="bird200_ViT-B-16_lr_0.1_decay_0_bsz_512_test_I20_warm", type=str,
                        help='which classifier to load')
    parser.add_argument('--feat_dim', type=int, default=512,
                        help='feat dim； 512 for ViT-B and 768 for ViT-L')
    # detection setting
    parser.add_argument('--score', default='MIP', type=str, choices=[
        'MCM', 'Maha', 'energy', 'max-logit', 'entropy', 'var', 'scaled', 'MSP'
    ], help='score options')

    # for MIP variants score
    parser.add_argument(
        '--template', default=['subset1'], type=str, choices=['full', 'subset1', 'subset2'])
    args = parser.parse_args()

    args.n_cls = get_num_cls(args)

    if args.server in ['inst-01', 'inst-03', 'inst-04']:
        args.root_dir = '/nobackup/dataset_myf'  # save dir of dataset
        # save dir of linear classsifier
        args.save_dir = f'/nobackup/checkpoints/clip_linear/{args.in_dataset}'
    elif args.server in ['galaxy-01', 'galaxy-02']:
        args.root_dir = '/nobackup-slow/dataset'
        # save dir of linear classsifier
        args.save_dir = f'/nobackup/checkpoints/clip_linear/{args.in_dataset}'
    elif args.server in ['A100']:
        args.root_dir = ''

    if args.in_dataset == 'ImageNet-subset':
        args.log_directory = f"results/ImageNet{args.num_imagenet_cls}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}_normalize_{args.normalize}"
    if args.in_dataset == 'ImageNet-dogs':
        args.log_directory = f"results/ImageNetDogs_{args.num_imagenet_cls}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}_normalize_{args.normalize}"
    else:
        args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}_normalize_{args.normalize}"
    if args.score == 'knn':
        args.log_directory += f'_k_{args.K}'
    os.makedirs(args.log_directory, exist_ok=True)

    return args


def get_test_labels(args, loader=None):
    if args.in_dataset in ['CIFAR-10', 'CIFAR-100']:
        test_labels = obtain_cifar_classes(
            root=args.root_dir, which_cifar=args.in_dataset)
    elif args.in_dataset == "ImageNet":
        test_labels = obtain_ImageNet_classes(
            loc=os.path.join('data', 'ImageNet'), option='clean')
    elif args.in_dataset == "ImageNet10":
        test_labels = obtain_ImageNet10_classes()
    elif args.in_dataset == "ImageNet20":
        test_labels = obtain_ImageNet20_classes()
    elif args.in_dataset == "ImageNet30":
        test_labels = obtain_ImageNet30_classes()
    elif args.in_dataset == "ImageNet100":
        test_labels = obtain_ImageNet100_classes(
            loc=os.path.join('./data', 'ImageNet100'))

    return test_labels


def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    torch.cuda.set_device(args.gpu)
    args.device = 'cuda'
    if args.model == 'CLIP':
        net, preprocess = set_model_clip(args)
    elif args.model == "CLIP-Linear":  # fine-tuned CLIP (linear layer only)
        net, preprocess = set_model_clip(args)
        args.ckpt = os.path.join(
            args.save_dir, f'{args.classifier_ckpt}_linear_probe_epoch_{args.epoch}.pth')
        linear_probe_dict = torch.load(
            args.ckpt,  map_location='cpu')['classifier']
        classifier = LinearClassifier(
            feat_dim=args.feat_dim, num_classes=args.n_cls)
        classifier.load_state_dict(linear_probe_dict)
        classifier.cuda()
        classifier.eval()
    elif args.model == 'vit':
        net, preprocess = set_model_vit()
    elif args.model == 'vit-Linear':
        net, preprocess = set_model_vit()
        args.ckpt = os.path.join(
            args.save_dir, f'{args.classifier_ckpt}_linear_probe_epoch_{args.epoch}.pth')
        linear_probe_dict = torch.load(
            args.ckpt,  map_location='cpu')['classifier']
        classifier = LinearClassifier(
            feat_dim=args.feat_dim, num_classes=args.n_cls)
        classifier.load_state_dict(linear_probe_dict)
        classifier.cuda()
        classifier.eval()

    net.eval()

    if args.in_dataset == 'CIFAR-10':
        log.debug('\nUsing CIFAR-10 as typical data')
        out_datasets = ['places365', 'SVHN',
                        'iSUN', 'dtd', 'LSUN', 'CIFAR-100']
    elif args.in_dataset == 'CIFAR-100':
        log.debug('\nUsing CIFAR-100 as typical data')
        out_datasets = ['places365', 'SVHN', 'iSUN', 'dtd', 'LSUN', 'CIFAR-10']
    elif args.in_dataset in ['ImageNet',  'ImageNet100', 'car196', 'flower102', 'food101', 'pet37']:
        out_datasets = ['SUN', 'places365', 'dtd', 'iNaturalist']
    elif args.in_dataset in ['ImageNet10']:
        out_datasets = ['ImageNet20']
    elif args.in_dataset in ['ImageNet20', 'ImageNet30']:
        out_datasets = ['ImageNet10']
    elif args.in_dataset == 'bird200':
        out_datasets = ['placesbg']

    if args.model in ['CLIP', 'CLIP-Linear', 'H-CLIP', 'H-CLIP-Linear']:
        test_loader = set_val_loader(args, preprocess)
        # used for KNN and Maha score
        train_loader = set_train_loader(
            args, preprocess, subset=args.subset)
    elif args.model in ['vit', 'vit-Linear', 'vit-Linear-H']:
        test_loader = set_val_loader_vit(args, preprocess)
        train_loader = set_train_loader_vit(
            args, preprocess, subset=args.subset)  # used for KNN and Maha score

    test_labels = get_test_labels(args, test_loader)

    if args.score == 'Maha':
        if args.generate:
            # this is faster than getting mean and var separately
            classwise_mean, precision = get_mean_prec(args, net, train_loader)

        classwise_mean = torch.load(os.path.join(
            args.template_dir, f'{args.model}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'), map_location='cpu').cuda()
        precision = torch.load(os.path.join(
            args.template_dir,  f'{args.model}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'), map_location='cpu').cuda()
        # args.normalize = True
        in_score = get_Mahalanobis_score(
            args, net, test_loader, classwise_mean, precision, in_dist=True)
    else:
        if args.model == 'CLIP':  # MIP and variants
            in_score, right_score, wrong_score = get_ood_scores_clip(
                args, net, test_loader, test_labels, in_dist=True)
        # after linear probe; img encoder -> logit space
        elif args.model in ['CLIP-Linear', 'vit-Linear']:
            in_score, right_score, wrong_score = get_ood_scores_clip_linear(
                args, net, classifier, test_loader, in_dist=True)
        num_right = len(right_score)
        num_wrong = len(wrong_score)
        log.debug('Error Rate {:.2f}'.format(
            100 * num_wrong / (num_wrong + num_right)))

    log.debug('\n\nError Detection')

    with open(f'score_T_{args.T}_{args.in_dataset}.npy', 'wb') as f:
        np.save(f, in_score)
    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        if args.in_dataset in ['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet30', 'ImageNet100', 'bird200', 'car196', 'flower102', 'food101', 'pet37']:
            ood_loader = set_ood_loader_ImageNet(
                args, out_dataset, preprocess, root=os.path.join(args.root_dir, 'ImageNet_OOD_dataset'))
        else:  # for CIFAR
            ood_loader = set_ood_loader(
                args, preprocess, out_dataset, preprocess)

        if args.score == 'Maha':
            out_score = get_Mahalanobis_score(
                args, net, ood_loader, classwise_mean, precision, in_dist=False)
        else:  # non knn scores
            if args.model in ['CLIP', 'H-CLIP']:
                out_score = get_ood_scores_clip(
                    args, net, ood_loader, test_labels)
            elif args.model in ['CLIP-Linear', 'vit-Linear']:
                out_score = get_ood_scores_clip_linear(
                    args, net, classifier, ood_loader)
        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        # debug
        # with open(f'score_T_{args.T}_{out_dataset}.npy', 'wb') as f:
        #     np.save(f, out_score)
        # end
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score,
                              auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                   np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)


if __name__ == '__main__':
    main()
