import os
import torch
import pandas as pd
from torchvision import datasets
import torchvision.transforms as transforms

# def build_caption_dataframe(params, dataset_name):
#     captions_path = os.path.join(params.captions_dir, "processed_captions_val2014.csv")
#     df = pd.read_csv(f"{captions_path}")
#     return df

# def build_image_loader(params, df, option):
#     image_ids = df["image_id"].values
#     image_prefix = f'COCO_{option}2014_'
#     image_filenames = [f"{image_path}/{image_prefix}{str(image_ids[i]).zfill(12)}.jpg" for i in range(len(image_ids))] 
    
#     dataset  = ImageFolderWithCaptions(data_dir, transform = preprocess)
#     dataset = CLIPDataset_ViT(params, image_filenames, df)
#     dataloader = torch.utils.data.DataLoader(dataset, batch_size=params.batch_size, num_workers=params.num_workers, shuffle=(option == 'train'))
#     return dataloader

class ImageFolderWithNames(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """
    def __init__(self, caption_df):
        self.caption_df = caption_df

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithNames, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        file_name = path.split("/")[-1]
        file_name_no_ext = file_name.split(".")[0]

        tuple_with_name = (original_tuple + (file_name_no_ext,))
        return tuple_with_name

normalize = transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                                        std=(0.26862954, 0.26130258, 0.27577711)) # for CLIP
preprocess = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
            ])

# EXAMPLE USAGE:
# instantiate the dataset and dataloader
data_dir = "/nobackup/dataset_myf/ImageNet10/val"
dataset = ImageFolderWithNames(data_dir, transform = preprocess) # our custom dataset
dataloader = torch.utils.data.DataLoader(dataset)

# iterate over data
for inputs, labels, paths in dataloader:
    # use the above variables freely
    # print(inputs, labels, paths)
    print(paths)