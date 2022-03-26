import os, sys
import argparse
import numpy as np
import torch
import clip
from scipy import stats
from models.linear import LinearClassifier
# from torchvision.transforms import transforms
from utils.common import obtain_ImageNet100_classes, obtain_ImageNet10_classes, obtain_ImageNet_classes, obtain_cifar_classes, setup_seed
from utils.detection_util import *
from utils.file_ops import prepare_dataframe, save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import set_model, set_train_loader, set_val_loader
# sys.path.append(os.path.dirname(__file__))

def process_args():
    parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #dataset
    parser.add_argument('--in_dataset', default='ImageNet', type=str, 
                        choices = ['CIFAR-10', 'CIFAR-100', 'ImageNet', 'ImageNet10', 'ImageNet100'], help='in-distribution dataset')
    parser.add_argument('-b', '--batch-size', default=500, type=int,
                            help='mini-batch size')
    #encoder loading
    parser.add_argument('--model', default='CLIP', choices = ['CLIP','CLIP-Linear'], type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'RN50x4', 'ViT-L/14'], help='which pretrained img encoder to use')
    #classifier loading
    parser.add_argument('--epoch', default ="40", type=str,
                             help='which epoch to test')
    parser.add_argument('--classifier_ckpt', default ="ImageNet_ViT-B-16_lr_0.5_decay_0_bsz_512_test_02_warm", type=str,
                             help='which classifier to load')
    parser.add_argument('--feat_dim', type=int, default=512, help='feat dim； 512 for ImageNet')
    #detection setting 
    parser.add_argument('--score', default='Maha', type=str, help='score options: Maha|MIP|MSP|energy|knn|MIPCT|MIPCI|retrival|MIPT|analyze')
    parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
    parser.add_argument('--T', default = 1, type =float, help = "temperature for energy score")    
    parser.add_argument('--K', default = 100, type =int, help = "# of nearest neighbor")
    parser.add_argument('--normalize', action='store_true', help='whether use normalized features for Maha score')
    #Misc 
    parser.add_argument('--seed', default = 1, type =int, help = "random seed")
    parser.add_argument('--name', default = "clean", type =str, help = "unique ID for the run")    
    parser.add_argument('--server', default = "inst-01", type =str, 
                choices = ['inst-01', 'inst-04', 'A100', 'galaxy-01', 'galaxy-02'], help = "on which server the experiment is conducted")
    parser.add_argument('--gpu', default=1, type=int,
                        help='the GPU indice to use')
    args = parser.parse_args()

    if args.in_dataset in ['CIFAR-10', 'ImageNet10']:
        args.n_cls = 10
    elif args.in_dataset in ['CIFAR-100', 'ImageNet100']:
        args.n_cls = 100
    elif args.in_dataset == "ImageNet":
        args.n_cls = 1000
    
    if args.server in ['inst-01', 'inst-04']:
        args.root_dir = '/nobackup/dataset_myf' #save dir of dataset
        args.save_dir = f'/nobackup/checkpoints/clip_linear/{args.in_dataset}' # save dir of linear classsifier
    elif args.server in ['galaxy-01', 'galaxy-02']:
        args.root_dir = '/nobackup-slow/dataset'
    elif args.server in ['A100']:
        args.root_dir = ''

    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok= True)

    return args

def get_test_labels(args):
    if args.in_dataset in  ['CIFAR-10', 'CIFAR-100']:
        test_labels = obtain_cifar_classes(root = args.root_dir, which_cifar = args.in_dataset)
    elif args.in_dataset ==  "ImageNet":
        test_labels = obtain_ImageNet_classes(loc = os.path.join('data','ImageNet'), option = 'clean')
    elif args.in_dataset ==  "ImageNet10":
        test_labels = obtain_ImageNet10_classes()
    elif args.in_dataset ==  "ImageNet100":
        test_labels = obtain_ImageNet100_classes(loc = os.path.join(os.path.join(args.root_dir,'ImageNet100')))
    return test_labels


def main():
    args = process_args()
    setup_seed(args)
    log = setup_log(args)
    torch.cuda.set_device(args.gpu)
    args.device = 'cuda'
    if args.model == 'resnet34': #not available now
        args.ckpt = f"/nobackup/checkpoints/{args.in_dataset}/{args.name}/checkpoint_{args.epoch}.pth.tar"
        pretrained_dict= torch.load(args.ckpt,  map_location='cpu')['state_dict']
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        net = set_model(args)
        net.load_state_dict(pretrained_dict)
    elif args.model == "CLIP": #pre-trained CLIP
        net, preprocess = clip.load(args.CLIP_ckpt, args.gpu) 
    elif args.model == "CLIP-Linear": #fine-tuned CLIP (linear layer only)
        net, preprocess = clip.load(args.CLIP_ckpt, args.gpu) 
        args.ckpt = os.path.join(args.save_dir, f'{args.classifier_ckpt}_linear_probe_epoch_{args.epoch}.pth')
        linear_probe_dict= torch.load(args.ckpt,  map_location='cpu')['classifier']
        classifier = LinearClassifier(feat_dim=args.feat_dim, num_classes=args.n_cls)
        classifier.load_state_dict(linear_probe_dict)
        classifier.cuda()
        classifier.eval()


    net.eval()
    test_labels = get_test_labels(args)
    if args.score in ['MIPCI', 'MIPCT', 'retrival']:
        captions_dir = 'gen_captions'
        text_df = prepare_dataframe(captions_dir, dataset_name = 'imagenet_val') # currently only supports ImageNet10 captions
        if args.score == 'MIPCT':
            in_score = get_MIPC_scores_clip(args, net, text_df, test_labels, in_dist=True)
        elif args.score == 'MIPCI':
            in_score = get_retrival_scores_from_classwise_mean_clip(args, net, text_df, preprocess)
        elif args.score == 'retrival':
            in_score = get_retrival_scores_clip(args, net, text_df, 12, num_per_cls = 10, generate = False, template_dir = 'img_templates')
    else:
        test_loader = set_val_loader(args, preprocess, root=args.root_dir)
        train_loader = set_train_loader(args, preprocess, root=args.root_dir) # used for KNN and Maha score
    # ood_num_examples = len(test_loader.dataset) 
    if args.score == 'analyze':
        analysis_feature_manitude(args, net, preprocess, test_loader) 
        return 

    if args.score in ['MIP', 'energy', 'entropy', 'MIPT', 'MSP', 'energy_logits']:
        if args.model == 'CLIP':
            in_score, right_score, wrong_score= get_ood_scores_clip(args, net, test_loader, test_labels, in_dist=True)
        elif args.model == 'CLIP-Linear':
            in_score, right_score, wrong_score= get_ood_scores_clip_linear(args, net, classifier, test_loader, in_dist=True)       
        num_right = len(right_score)
        num_wrong = len(wrong_score)
        log.debug('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))
    elif args.score == 'knn':
        in_score, index_bad = get_knn_scores_from_clip_img_encoder_id(args, net, train_loader, test_loader)
    elif args.score == 'Maha':
        # mean = get_mean(args, net, preprocess)
        # prec = get_prec(args, net, train_loader)
        generate = True
        template_dir = 'img_templates'
        if generate: 
            classwise_mean, precision = get_mean_prec(args, net, preprocess) # this is faster than getting mean and var separately
        else: 
            classwise_mean = torch.load(os.path.join(template_dir,f'classwise_mean_{args.in_dataset}.pt'), map_location= 'cpu').cuda()
            precision = torch.load(os.path.join(template_dir,f'precision_{args.in_dataset}.pt'), map_location= 'cpu').cuda()
        in_score = get_Mahalanobis_score(args, net, test_loader, classwise_mean, precision, in_dist = True)

    if args.in_dataset == 'CIFAR-10':
        log.debug('\nUsing CIFAR-10 as typical data') 
        out_datasets = ['places365','SVHN', 'iSUN', 'dtd', 'LSUN', 'CIFAR-100']
    elif args.in_dataset == 'CIFAR-100': 
        log.debug('\nUsing CIFAR-100 as typical data')
        # out_datasets = [ 'SVHN', 'places365','LSUN_resize', 'iSUN', 'dtd', 'LSUN', 'cifar10']
        out_datasets =  ['places365','SVHN', 'iSUN', 'dtd', 'LSUN', 'CIFAR-10']
    elif args.in_dataset in ['ImageNet','ImageNet10', 'ImageNet100']: 
        out_datasets =  ['places365','SUN', 'dtd', 'iNaturalist']
        # out_datasets =  ['places365', 'dtd', 'iNaturalist']
    log.debug('\n\nError Detection')

    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        # if caption as input
        if args.score == 'MIPCT':
            ood_text_df = prepare_dataframe(captions_dir, dataset_name = out_dataset) 
            out_score = get_MIPC_scores_clip(args, net, ood_text_df, test_labels)
        elif args.score == 'MIPCI':
            ood_text_df = prepare_dataframe(captions_dir, dataset_name = out_dataset) 
            out_score = get_retrival_scores_from_classwise_mean_clip(args, net, ood_text_df, preprocess)
        elif args.score == 'retrival':
            ood_text_df = prepare_dataframe(captions_dir, dataset_name = out_dataset) 
            out_score = get_retrival_scores_clip(args, net, ood_text_df, preprocess, num_per_cls = 10, generate = False, template_dir = 'img_templates')
        else: # image as input 
            if args.in_dataset in ['ImageNet', 'ImageNet10', 'ImageNet100']:
                ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root= os.path.join(args.root_dir,'ImageNet_OOD_dataset'))
            else: #for CIFAR
                ood_loader = set_ood_loader(args, out_dataset, preprocess)
            if args.score == 'knn':
                out_score = get_knn_scores_from_clip_img_encoder_ood(args, net, ood_loader, index_bad)
            else: # non knn scores
                if args.model == 'CLIP':
                    out_score = get_ood_scores_clip(args, net, ood_loader, test_labels) 
                elif args.model == 'CLIP-Linear':
                    out_score = get_ood_scores_clip_linear(args, net, classifier, ood_loader) 
        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)

if __name__ == '__main__':
    main()