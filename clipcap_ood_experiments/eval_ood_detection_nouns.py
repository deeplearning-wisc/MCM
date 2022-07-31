import os, sys
import argparse
import numpy as np
import torch
import clip
from scipy import stats
# from torchvision.transforms import transforms
from utils.common import *
from utils.detection_util import get_and_print_results, print_measures, set_ood_loader_ImageNet
from utils.nouns_util import *
from utils.file_ops import  save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
# sys.path.append(os.path.dirname(__file__))

def get_num_cls(args):    
    NUM_CLS_DICT = {
        'CIFAR-10': 10, 'ImageNet10': 10, 'ImageNet10_original': 10,
        'ImageNet20': 20, 'ImageNet30': 30,
    }
    if args.in_dataset in ['ImageNet-subset', 'ImageNet-dogs']:
        n_cls = args.num_imagenet_cls
    else:
        n_cls = NUM_CLS_DICT[args.in_dataset]
    return n_cls

def process_args():
    parser = argparse.ArgumentParser(description='Evaluates a CIFAR OOD Detector',
                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    #unique setting for each run
    parser.add_argument('--in_dataset', default='ImageNet10', type=str, 
                        choices = ['CIFAR-10', 'CIFAR-100',  
                        'ImageNet10_original', 'ImageNet10', 'ImageNet20'], help='in-distribution dataset')
    parser.add_argument('--name', default = "test_I20_new_str_filter", type =str, help = "unique ID for the run")
    #test_imagenet100_10_seed_1  
    parser.add_argument('--seed', default = 4, type =int, help = "random seed")  
    parser.add_argument('--server', default = 'inst-01', type =str, 
                choices = ['inst-01', 'inst-04', 'A100', 'galaxy-01', 'galaxy-02'], help = "on which server the experiment is conducted")
    parser.add_argument('--gpu', default=5, type=int, help='the GPU indice to use')
    # batch size. num of classes
    parser.add_argument('-b', '--batch-size', default=1, type=int,
                            help='mini-batch size; 1 for nouns score; 75 for odin_logits; 512 for other scores [clip]')
    #encoder loading
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'RN50x4', 'ViT-L/14'], help='which pretrained img encoder to use')
    parser.add_argument('--feat_dim', type=int, default=512, help='feat dim； 512 for ViT-B and 768 for ViT-L')
    #detection setting  
    parser.add_argument('--score', default='clipcap_nouns', type=str, choices = ['clipcap_nouns', 'ofa_nouns'], help='score options')  
    # for ODIN score 
    parser.add_argument('--T', default = 1, type =float, help = "temperature") 
    # for fingerprint score 
    parser.add_argument('--softmax', type = bool, default = False, help='whether to apply softmax to the inner prod')
    #Misc 
    parser.add_argument('--out_as_pos', action='store_true', help='OE define OOD data as positive.')
    #for MIP variants score
    args = parser.parse_args()

    args.n_cls = get_num_cls(args)
    
    if args.server in ['inst-01', 'inst-03', 'inst-04']:
        args.root_dir = '/nobackup/dataset_myf' #save dir of dataset
        args.save_dir = f'/nobackup/checkpoints/clip_linear/{args.in_dataset}' # save dir of linear classsifier
    elif args.server in ['galaxy-01', 'galaxy-02']:
        args.root_dir = '/nobackup-slow/dataset'
        args.save_dir = f'/nobackup/checkpoints/clip_linear/{args.in_dataset}' # save dir of linear classsifier
    elif args.server in ['A100']:
        args.root_dir = ''

    args.log_directory = f"nouns_results/{args.in_dataset}/{args.score}/{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok= True)

    return args

def obtain_ImageNet20_classes(loc = None):

    class_dict =   {"n04147183": "sailboat", "n02951358": "canoe" , "n02782093": "balloon", "n04389033": "tank", "n03773504": "missile",
    "n02917067": "bullet train", "n02317335": "starfish", "n01632458":"spotted salamander", "n01630670":"common newt", "n01631663": "zebra",
    "n02391049": "frilled lizard", "n01693334":"green lizard", "n01697457": "African crocodile", "n02120079": "Arctic fox", "n02114367": "timber wolf",  
    "n02132136": "brown bear", "n03785016": "moped", "n04310018": "steam locomotive", "n04266014": "space shuttle", "n04252077": "snowmobile"}
    # sort by values
    class_dict =  {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[0])}
    return class_dict.values()

def obtain_ImageNet10_classes(original = True):
    if original:
        class_dict = {'plane': 'n04552348', 'car': 'n04285008', 'bird': 'n01530575', 'cat':'n02123597', 
            'antelope' : 'n02422699', 'dog':'n02107574', 'frog':'n01641577',  'snake':'n01728572', 
            'ship':'n03095699', 'truck':'n03417042'}
    else:
        class_dict =   {"warplane": "n04552348", "sports car":"n04285008", 
            'brambling bird':'n01530575', "Siamese cat": 'n02123597', 
            'antelope': 'n02422699', 'swiss mountain dog':'n02107574',
            "bull frog":"n01641577", 'garbage truck':"n03417042",
            "horse" :"n02389026", "container ship": "n03095699"}
        # class_dict =   {'aircraft': "n04552348", "car":"n04285008", 
        #     'bird':'n01530575', "cat": 'n02123597', 
        #     'antelope': 'n02422699', 'dog':'n02107574',
        #     "frog":"n01641577", 'truck':"n03417042",
        #     "horse" :"n02389026", "ship": "n03095699"}
    # sort by values
    class_dict =  {k: v for k, v in sorted(class_dict.items(), key=lambda item: item[1])}
    return class_dict.keys()

def main():
    args = process_args()
    setup_seed(args)
    log = setup_log(args)
    torch.cuda.set_device(args.gpu)
    args.device = 'cuda'
    net, preprocess = clip.load(args.CLIP_ckpt, args.gpu) 

    net.eval()

    if args.in_dataset == 'CIFAR-10':
        log.debug('\nUsing CIFAR-10 as typical data') 
        out_datasets = ['places365','SVHN', 'iSUN', 'dtd', 'LSUN', 'CIFAR-100']
    elif args.in_dataset == 'CIFAR-100': 
        log.debug('\nUsing CIFAR-100 as typical data')
        # out_datasets = [ 'SVHN', 'places365','LSUN_resize', 'iSUN', 'dtd', 'LSUN', 'cifar10']
        out_datasets =  ['places365','SVHN', 'iSUN', 'dtd', 'LSUN', 'CIFAR-10']
    elif args.in_dataset in [ 'ImageNet10', 'ImageNet10_original', 'ImageNet20']: 
        out_datasets =  ['SUN', 'places365','dtd', 'iNaturalist']
        # out_datasets = ['ImageNet10']

    test_loader = set_val_loader(args, preprocess)  
    if args.in_dataset ==  "ImageNet10_original":
        test_labels = obtain_ImageNet10_classes(original = True)
    elif args.in_dataset ==  "ImageNet10":
        test_labels = obtain_ImageNet10_classes(original = False)  
    elif args.in_dataset ==  "ImageNet20":
        test_labels = obtain_ImageNet20_classes()  

    
    in_score = get_nouns_scores_clip(args, preprocess, net, test_loader, list(test_labels), args.in_dataset, in_dist = True, filter = "str", debug = False)


    log.debug('\n\nError Detection')

    with open(f'score_T_{args.T}_{args.in_dataset}.npy', 'wb') as f:
            np.save(f, in_score)
    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        # if caption as input
        if args.in_dataset in [ 'ImageNet10','ImageNet10_original', 'ImageNet20']:
            ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root= os.path.join(args.root_dir,'ImageNet_OOD_dataset'))
        else: #for CIFAR
            ood_loader = set_ood_loader_ImageNet(args, preprocess, out_dataset, preprocess)

        out_score = get_nouns_scores_clip(args, preprocess, net, ood_loader, list(test_labels), out_dataset, in_dist = False, filter = "str", debug = False)

        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score, auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list), np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)

if __name__ == '__main__':
    main()