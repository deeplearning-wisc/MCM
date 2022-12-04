import os
import argparse
import numpy as np
import torch
from scipy import stats

from utils.common import setup_seed, get_num_cls, get_test_labels
from utils.detection_util import get_Mahalanobis_score, get_mean_prec, print_measures, get_and_print_results, get_ood_scores_clip
from utils.file_ops import save_as_dataframe, setup_log
from utils.plot_util import plot_distribution
from utils.train_eval_util import  set_model_clip, set_train_loader, set_val_loader, set_ood_loader_ImageNet
# sys.path.append(os.path.dirname(__file__))


def process_args():
    parser = argparse.ArgumentParser(description='Evaluates MCM Score for CLIP',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # setting for each run
    parser.add_argument('--in_dataset', default='ImageNet20', type=str,
                        choices=['ImageNet', 'ImageNet10', 'ImageNet20', 'ImageNet100',
                                  'pet37', 'food101', 'car196', 'bird200'], help='in-distribution dataset')
    parser.add_argument('--root-dir', default="datasets", type=str,
                        help='root dir of datasets')
    parser.add_argument('--name', default="release_test",
                        type=str, help="unique ID for the run")
    parser.add_argument('--seed', default=5, type=int, help="random seed")
    parser.add_argument('--gpu', default=0, type = int,
                        help='the GPU indice to use')
    parser.add_argument('-b', '--batch-size', default=512, type=int,
                        help='mini-batch size')
    parser.add_argument('--T', type=int, default=1,
                        help='temperature parameter')
    parser.add_argument('--model', default='CLIP', type=str, help='model architecture')
    parser.add_argument('--CLIP_ckpt', type=str, default='ViT-B/16',
                        choices=['ViT-B/32', 'ViT-B/16', 'ViT-L/14'], help='which pretrained img encoder to use')
    parser.add_argument('--score', default='MCM', type=str, choices=[
        'MCM', 'energy', 'max-logit', 'entropy', 'var', 'maha'], help='score options')
    # for Mahalanobis score
    parser.add_argument('--normalize', type = bool, default = False, help='whether use normalized features for Maha score')
    parser.add_argument('--generate', type = bool, default = True, help='whether to generate class-wise means or read from files for Maha score')
    parser.add_argument('--template_dir', type = str, default = 'img_templates', help='the loc of stored classwise mean and precision matrix')
    parser.add_argument('--subset', default = True, type =bool, help = "whether uses a subset of samples in the training set")
    parser.add_argument('--max_count', default = 250, type =int, help = "how many samples are used to estimate classwise mean and precision matrix")
    args = parser.parse_args()

    args.n_cls = get_num_cls(args)
    args.log_directory = f"results/{args.in_dataset}/{args.score}/{args.model}_{args.CLIP_ckpt}_T_{args.T}_ID_{args.name}"
    os.makedirs(args.log_directory, exist_ok=True)

    return args

def main():
    args = process_args()
    setup_seed(args.seed)
    log = setup_log(args)
    assert torch.cuda.is_available()
    torch.cuda.set_device(args.gpu)

    net, preprocess = set_model_clip(args)
    net.eval()

    if args.in_dataset in ['ImageNet10']: 
        out_datasets = ['ImageNet20']
    elif args.in_dataset in ['ImageNet20']: 
        out_datasets = ['ImageNet10']
    elif args.in_dataset in [ 'ImageNet', 'ImageNet100', 'bird200', 'car196', 'food101', 'pet37']:
         out_datasets = ['iNaturalist','SUN', 'places365', 'dtd']
    test_loader = set_val_loader(args, preprocess)
    test_labels = get_test_labels(args, test_loader)

    if args.score == 'maha':
        os.makedirs(args.template_dir, exist_ok = True)
        train_loader = set_train_loader(args, preprocess, subset = args.subset) 
        if args.generate: 
            classwise_mean, precision = get_mean_prec(args, net, train_loader)
        classwise_mean = torch.load(os.path.join(args.template_dir, f'{args.model}_classwise_mean_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'), map_location= 'cpu').cuda()
        precision = torch.load(os.path.join(args.template_dir,  f'{args.model}_precision_{args.in_dataset}_{args.max_count}_{args.normalize}.pt'), map_location= 'cpu').cuda()
        in_score = get_Mahalanobis_score(args, net, test_loader, classwise_mean, precision, in_dist = True)
    else:
        in_score  = get_ood_scores_clip(args, net, test_loader, test_labels, in_dist=True)

    auroc_list, aupr_list, fpr_list = [], [], []
    for out_dataset in out_datasets:
        log.debug(f"Evaluting OOD dataset {out_dataset}")
        ood_loader = set_ood_loader_ImageNet(args, out_dataset, preprocess, root=os.path.join(args.root_dir, 'ImageNet_OOD_dataset'))
        if args.score == 'maha':
            out_score = get_Mahalanobis_score(args, net, ood_loader, classwise_mean, precision, in_dist = False)
        else:
            out_score = get_ood_scores_clip(args, net, ood_loader, test_labels)
        log.debug(f"in scores: {stats.describe(in_score)}")
        log.debug(f"out scores: {stats.describe(out_score)}")
        plot_distribution(args, in_score, out_score, out_dataset)
        get_and_print_results(args, log, in_score, out_score,
                              auroc_list, aupr_list, fpr_list)
    log.debug('\n\nMean Test Results')
    print_measures(log, np.mean(auroc_list), np.mean(aupr_list),
                   np.mean(fpr_list), method_name=args.score)
    save_as_dataframe(args, out_datasets, fpr_list, auroc_list, aupr_list)


if __name__ == '__main__':
    main()
