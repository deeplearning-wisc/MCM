import os
import shutil 

def create_ImageNet_subset(src, dst, target_dirs):
    assert(os.path.exists(src))
    if not os.path.exists(dst):
        os.makedirs(dst)
    types = ['train', 'val']
    for type in types:
        for dir_name in os.listdir(os.path.join(src, type)):
            if dir_name in target_dirs:
                shutil.copytree(os.path.join(src, type, dir_name), os.path.join(dst,type, dir_name))
    

if __name__ == '__main__':
    class_dict = {'plane': 'n04552348', 'car': 'n04285008', 'bird': 'n01530575', 'cat':'n02123597', 
        'antelope' : 'n02422699', 'dog':'n02107574', 'frog':'n01641577',  'snake':'n01728572', 
        'ship':'n03095699', 'truck':'n03417042'}
    #create on inst-01
    create_ImageNet_subset(src = '/nobackup/ImageNet', dst = '/nobackup/dataset_myf/ImageNet10', target_dirs = class_dict.values())