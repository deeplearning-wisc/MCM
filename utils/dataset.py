import os
import cv2
import torch
import numpy as np
import pandas as pd
import torchvision
from PIL import Image
from transformers import ViTFeatureExtractor, CLIPProcessor

def load_train_data(params):
    train_df  = prepare_dataframe(params.lang, params.captions_path, True)
    train_loader = build_loaders(train_df, "train", params)
    return train_loader

def load_val_data(params):
    train_df, valid_df = prepare_dataframe(params.lang, params.captions_path, False)
    valid_loader = build_loaders(valid_df, "valid", params)
    return valid_loader

def prepare_dataframe(lang, captions_path, train_only = True):
    # load caption file
    if lang == 'es':
        df = pd.read_csv(f"{captions_path}", encoding = 'utf-8-sig')
    else:
        df = pd.read_csv(f"{captions_path}")

    x = list(set(df['image_id'].values))
    image_ids = np.arange(0, len(x))
    if train_only:
        train_images = [x[i] for i in image_ids]
        train_df = df[df["image_id"].isin(train_images)].reset_index(drop=True)
        return train_df
    else:
        np.random.seed(42)
        valid_ids = np.random.choice(
            image_ids, size=int(0.1 * len(image_ids)), replace=False
        )
        # valid_ids = image_ids[len(image_ids)-1000:len(image_ids)]
        train_ids = [id_ for id_ in image_ids if id_ not in valid_ids]
        train_images = [x[i] for i in train_ids]
        val_images = [x[i] for i in valid_ids]
        train_df = df[df["image_id"].isin(train_images)].reset_index(drop=True)
        valid_df = df[df["image_id"].isin(val_images)].reset_index(drop=True)
        return train_df, valid_df


def build_loaders(df, mode, params):
    image_ids = df["image_id"].values
    image_filenames = [f"{params.image_path}/{params.image_prefix}{str(image_ids[i]).zfill(12)}.jpg" for i in range(len(image_ids))] 
    
    dataset = CLIPDataset_ViT(
        image_filenames,
        df["caption"].values,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=params.batch_size,
        num_workers=params.num_workers,
        shuffle=True,
    )
    return dataloader

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
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = np.moveaxis(image, source=-1, destination=0)
        inputs = self.feature_extractor(images = img, return_tensors="pt") # transforms already

        item = {}
        item['image'] = inputs['pixel_values'][0]  # inputs['pixel_values'].shape: torch.Size([1, 3, 224, 224])
        item['caption'] = self.captions[idx]
        return item


    def __len__(self):
        return len(self.captions)