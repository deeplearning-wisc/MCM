import os, sys
import logging
import argparse
import numpy as np
import torch
import clip
from torchvision.transforms import transforms
from utils.common import obtain_ImageNet10_classes, obtain_ImageNet_classes, obtain_cifar_classes
from utils.detection_util import get_and_print_results, get_ood_scores, get_ood_scores_clip, print_measures, set_ood_loader, set_ood_loader_ImageNet
from utils.train_eval_util import set_model, set_val_loader
# sys.path.append(os.path.dirname(__file__))

def process_args():
    parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default="ImageNet10", type=str, choices = ['CIFAR-10', 'CIFAR-100', 'ImageNet', 'ImageNet10'], help='in-distribution dataset')
    parser.add_argument('--gpus', default=[4], nargs='*', type=int,
                            help='List of GPU indices to use, e.g., --gpus 0 1 2 3')
    parser.add_argument('-b', '--batch-size', default=400, type=int,
                            help='mini-batch size')
    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'RN50x4'], help='which pretrained img encoder to use')
    parser.add_argument('--name', default = "test_32_new_label", type =str, help = "name of the run to be tested")

    parser.add_argument('--epoch', default ="", type=str,
                            help='which epoch to test')
        
    parser.add_argument('--score', default='MSP', type=str, help='score options: MSP|energy|')
    parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
    parser.add_argument('--use_xent', '-x', action='store_true', help='Use cross entropy scoring instead of the MSP.')
    
    parser.add_argument('--seed', default = 1, type =int, help = "random seed")
    args = parser.parse_args()

    args.gpus = list(map(lambda x: torch.device('cuda', x), args.gpus)) # will be used in set_model()

    if args.in_dataset == "CIFAR-10":
        args.n_cls = 10
    elif args.in_dataset == "CIFAR-100":
        args.n_cls = 100
    elif args.in_dataset == "ImageNet":
        args.n_cls = 1000
    elif args.in_dataset == "ImageNet10":
        args.n_cls = 10
    return args

def get_test_labels(args):
    if args.in_dataset in  ['CIFAR-10', 'CIFAR-100']:
        test_labels = obtain_cifar_classes(root = '/nobackup/dataset_myf', which_cifar = args.in_dataset)
    elif args.in_dataset ==  "ImageNet":
        # imagenet_cls = obtain_ImageNet_classes(loc = os.path.join('data','imagenet_class_index.json')) # uncleaned labels
        test_labels = obtain_ImageNet_classes(loc = os.path.join('data','imagenet_class_clearn.npy'), cleaned = True)
    elif args.in_dataset ==  "ImageNet10":
        test_labels = obtain_ImageNet10_classes()
    return test_labels

def setup_log(args):
    if args.score  in ['MSP']:
        args.log_directory = "results/{in_dataset}/{name}/{score}_new_debug".\
                        format(in_dataset=args.in_dataset, name= args.name, score = args.score)
    if not os.path.exists(args.log_directory):
        os.makedirs(args.log_directory)
    
    log = logging.getLogger(__name__)
    formatter = logging.Formatter('%(asctime)s : %(message)s')
    fileHandler = logging.FileHandler(os.path.join(args.log_directory, "ood_eval_info.log"), mode='w')
    fileHandler.setFormatter(formatter)
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(formatter)
    log.setLevel(logging.DEBUG)
    log.addHandler(fileHandler)
    log.addHandler(streamHandler) 
    log.debug(f"#########{args.name}############")
    return log

def setup_seed(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)

def save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list):
    fpr_list = [float('{:.2f}'.format(100*fpr)) for fpr in fpr_list]
    auroc_list = [float('{:.2f}'.format(100*auroc)) for auroc in auroc_list]
    aupr_list = [float('{:.2f}'.format(100*aupr)) for aupr in aupr_list]
    import pandas as pd
    data = {k:v for k,v in zip(out_datasets, zip(fpr_list,auroc_list,aupr_list))}
    # Specify orient='index' to create the DataFrame using dictionary keys as rows
    df = pd.DataFrame.from_dict(data, orient='index',
                       columns=['FPR95', 'AURPC', 'AUPR'])
    df.to_csv(os.path.join(args.log_directory,f'{args.name}.csv'))
        

def main():

    args = process_args()
    setup_seed(args)
    log = setup_log(args)
    if args.model == 'resnet34': #not available now
        args.ckpt = f"/nobackup/checkpoints/{args.in_dataset}/{args.name}/checkpoint_{args.epoch}.pth.tar"
        pretrained_dict= torch.load(args.ckpt,  map_location='cpu')['state_dict']
        pretrained_dict = {key.replace("module.", ""): value for key, value in pretrained_dict.items()}
        net = set_model(args)
        net.load_state_dict(pretrained_dict)
    elif args.model == "CLIP": #available option
        torch.cuda.set_device(args.gpus[0])
        net, preprocess = clip.load(args.CLIP_ckpt, args.gpus[0]) 

    net.eval()
    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args)
    # ood_num_examples = len(test_loader.dataset) 
    # num_batches = ood_num_examples // args.batch_size

    if args.score == 'MSP':
            if args.model == 'CLIP':
                in_score, right_score, wrong_score= get_ood_scores_clip(args, net, test_loader, test_labels, in_dist=True) 
            else:
                in_score, right_score, wrong_score= get_ood_scores(args, net, test_loader, in_dist=True)       
    num_right = len(right_score)
    num_wrong = len(wrong_score)
    log.debug('Error Rate {:.2f}'.format(100 * num_wrong / (num_wrong + num_right)))

    if args.in_dataset == 'CIFAR-10':
        log.debug('\nUsing CIFAR-10 as typical data') 
        out_datasets = ['places365','SVHN', 'iSUN', 'dtd', 'LSUN']
    elif args.in_dataset == 'CIFAR-100': 
        log.debug('\nUsing CIFAR-100 as typical data')
        # out_datasets = [ 'SVHN', 'places365','LSUN_resize', 'iSUN', 'dtd', 'LSUN', 'cifar10']
        out_datasets =  ['places365','SVHN', 'iSUN', 'dtd', 'LSUN']
    elif args.in_dataset in ['ImageNet','ImageNet10']: 
        out_datasets =  ['places365','SUN', 'dtd', 'iNaturalist']
    log.debug('\n\nError Detection')

    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        if args.in_dataset in ['ImageNet', 'ImageNet10']:
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess)
        else: #for CIFAR
            ood_loader = set_ood_loader(args, out_dataset, preprocess)
    
        if args.model == 'CLIP':
            out_score = get_ood_scores_clip(args, net, ood_loader, test_labels) 
        else:
            out_score = get_ood_scores(args, net, ood_loader)
        get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)

if __name__ == '__main__':
    main()