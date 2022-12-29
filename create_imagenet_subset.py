import os
import shutil
from tqdm import tqdm
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create ImageNet subset',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--in_dataset', default='ImageNet10', type=str,
                        choices=['ImageNet10', 'ImageNet20', 'ImageNet100'], help='in-distribution dataset')
    parser.add_argument('--src-dir', default='/nobackup/ImageNet', type=str,
                        help='full path of ImageNet-1k')
    parser.add_argument('--dst-dir', default='datasets_temp', type=str,
                        help='root dir of in_dataset')
    args = parser.parse_args()
    dst_path = os.path.join(args.dst_dir, f"{args.in_dataset}")
    os.makedirs(dst_path, exist_ok=True)

    for split in ['train', 'val']:
        with open(os.path.join('data',f'{args.in_dataset}','class_list.txt')) as file:
            for line in tqdm(file.readlines()):
                cls = line.strip()
                shutil.copytree(os.path.join(args.src_dir, split, cls), os.path.join(dst_path, split, cls), dirs_exist_ok=True)