from torch.utils.data import Dataset
import os
import numpy.random as npr
from PIL import Image

class ImageNetSubset(Dataset):

    def __init__(self, n_cls, root, class_list_loc='./data', id='', save=True, train=True, transform=None, target_transform=None, seed=0):
        npr.seed(seed)
        self.n_cls = int(n_cls)
        self.split = 'train' if train else 'val'
        self.class_list = list(npr.permutation(os.listdir(os.path.join(root, self.split)))[0:self.n_cls])
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.data = [(image_id, cls) \
            for cls in self.class_list \
            for image_id in os.listdir(os.path.join(root, self.split, cls))]
        print(f'ImageNet {self.split} subset with {len(self.class_list)} classes, {len(self.data)} samples')

        if save:
            save_path = os.path.join(class_list_loc, f'ImageNet{n_cls}', id)
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            with open(os.path.join(save_path, 'class_list.txt'), 'w+') as f:
                f.writelines([line + '\n' for line in self.class_list])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_id, cls = self.data[index]
        label = self.class_list.index(cls)
        image_path = os.path.join(self.root, self.split, cls, image_id)
        image = Image.open(image_path)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
