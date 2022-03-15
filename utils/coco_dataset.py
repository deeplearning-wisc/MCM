import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
import argparse
from transformers import ViTFeatureExtractor, CLIPProcessor


def build_caption_dataframe(lang, captions_dir, option = 'train'):
    if option == 'train': 
        image_path =os.path.join(params.image_dir, "train2014")
        captions_path = os.path.join(params.captions_dir, "processed_captions_train2014.csv")
    elif option == 'val':
        image_path =os.path.join(params.image_dir, "val2014")
        captions_path = os.path.join(params.captions_dir, "processed_captions_val2014.csv")
    if lang == 'es':
        df = pd.read_csv(f"{captions_path}", encoding = 'utf-8-sig')
    elif lang == 'en':
        df = pd.read_csv(f"{captions_path}")
    x = list(set(df['image_id'].values))
    image_ids = np.arange(0, len(x))
    images = [x[i] for i in image_ids]
    processed_df = df[df["image_id"].isin(images)].reset_index(drop=True)
    return processed_df, image_path

def build_image_loader(params, df, image_path, option):
    image_ids = df["image_id"].values
    image_prefix = f'COCO_{option}2014_'
    image_filenames = [f"{image_path}/{image_prefix}{str(image_ids[i]).zfill(12)}.jpg" for i in range(len(image_ids))] 
    
    dataset = CLIPDataset_ViT(image_filenames, df["caption"].values)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=(option == 'train'))
    return dataloader

def build_coco_loader(params, option = 'train'):
    df, image_path = build_caption_dataframe(params.lang, params.captions_dir, option)
    coco_loader = build_image_loader(params, df, image_path, option)
    return coco_loader


class CLIPDataset_ViT(torch.utils.data.Dataset):
    def __init__(self, image_filenames, captions, transforms=None):
        """
        image_filenames and cpations must have the same length; so, if there are
        multiple captions for each image, the image_filenames must have repetitive
        file names 
        """
        self.captions = list(captions)
        self.image_filenames = image_filenames
        # self.feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.feature_extractor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __getitem__(self, idx):
        image = cv2.imread(self.image_filenames[idx]) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # CV2 reads in img with BGR for historical reasons; need to conver to RGB first
        img = np.moveaxis(image, source=-1, destination=0)
        inputs = self.feature_extractor(images = img, return_tensors="pt") # transform to tensors already

        item = {}
        item['image'] = inputs['pixel_values'][0]  # inputs['pixel_values'].shape: torch.Size([1, 3, 224, 224])
        item['caption'] = self.captions[idx]
        return item


    def __len__(self):
        return len(self.captions)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test clip loader')
    parser.add_argument("--lang", type=str, default='en', help="Source language")
    parser.add_argument("--dataset", type=str, default='COCO', help="Source language")
    parser.add_argument("--caption_dir", type=str, default='data', help="Source language")

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    params = parser.parse_args()

    params.image_dir = f'/nobackup/COCO/COCO-14'
    params.captions_dir = f"{params.caption_dir}/{params.dataset}/captions/{params.lang}"
    train_loader = build_coco_loader(params, option = 'train')
    val_loader = build_coco_loader(params, option = 'val')