import torch
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import requests
import os, sys
import numpy as np
from transformers import CLIPProcessor, CLIPModel, CLIPTokenizer, CLIPTextModel
from torchvision.transforms import (CenterCrop, 
                                    Compose, 
                                    Normalize, 
                                    RandomHorizontalFlip,
                                    RandomResizedCrop, 
                                    Resize, 
                                    ToTensor)
# print(os.getcwd())
# print(sys.path)
from pathlib import Path
PROJ_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ_DIR))
from utils.common import evaluate_classification_huggingface, get_image_dataloader, obtain_cifar_classes

def play_with_image_processing(url =  "http://images.cocodataset.org/val2017/000000039769.jpg"): 
    image = Image.open(requests.get(url, stream=True).raw) #PIL (pillow) image;
    np_image = np.array(image) #shape -> (480, 640, 3); PIL or Numpy array represents image with channels in the last dimension
    transform = transforms.ToTensor()
    tensor = transform(np_image) # shape -> torch.Size([3, 480, 640])
    return image


def play_with_tokenizer_and_model(lang = 'ml'):
    from transformers import BertTokenizer, BertModel
    if lang == 'ml':
        tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')
        # text_inputs = tokenizer("今天真好")
        model = BertModel.from_pretrained("bert-base-multilingual-uncased")
        print(model.embeddings)
        #BertEmbeddings(
        # (word_embeddings): Embedding(105879, 768, padding_idx=0)
        # (position_embeddings): Embedding(512, 768)
        # (token_type_embeddings): Embedding(2, 768)
        # (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        # (dropout): Dropout(p=0.1, inplace=False)
        # )
    elif lang == 'es':
        tokenizer = BertTokenizer.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        model = BertModel.from_pretrained('dccuchile/bert-base-spanish-wwm-cased')
        print(model.embeddings)

        text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch16")
        #BertEmbeddings(
        #   (word_embeddings): Embedding(31002, 768, padding_idx=1)
        #   (position_embeddings): Embedding(512, 768)
        #   (token_type_embeddings): Embedding(2, 768)
        #   (LayerNorm): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
        #   (dropout): Dropout(p=0.1, inplace=False)
        # )
        print('done')

    elif lang == 'en':
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        text_inputs = tokenizer("Hello world")
        # text_inputs.keys() -> dict_keys(['input_ids', 'attention_mask'])
        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        print(model.text_model.embeddings)
        # CLIPTextEmbeddings(
        # (token_embedding): Embedding(49408, 512)
        # (position_embedding): Embedding(77, 512)
        # )
    print("")



def play_with_feature_extrator():
    from transformers import CLIPFeatureExtractor
    feature_extrator = CLIPFeatureExtractor.from_pretrained("openai/clip-vit-base-patch32")
    image = play_with_image_processing()
    image_inputs = feature_extrator(image)
    # 1. image_inputs.keys() -> dict_keys(['pixel_values'])
    # 2. image_inputs['pixel_values'][0].shape -> (3, 224, 224)
    print("")

def play_with_huggingface_clip(ckpt = "openai/clip-vit-base-patch32"):
    '''
    ref code: https://huggingface.co/transformers/v4.6.0/_modules/transformers/models/clip/modeling_clip.html#CLIPModel
    '''
    model = CLIPModel.from_pretrained(ckpt)
    for name, parameter in model.named_parameters(): 
        print(name, parameter.requires_grad)
        # note: 1. model.text_projection -> Linear(in_features=512, out_features=512, bias=False)
        #       2. model.visual_projection -> Linear(in_features=768, out_features=512, bias=False)
    processor = CLIPProcessor.from_pretrained(ckpt)
    image = play_with_image_processing()
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    # inputs.data.keys() -> dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
    # inputs.data['pixel_values'].shape -> torch.Size([1, 3, 224, 224])
    outputs = model(**inputs)
    CLS_feature_after_encoder = outputs.text_model_output.pooler_output # shape -> [2, 512]
    img_featuer_after_encoder = outputs.vision_model_output.pooler_output # shape -> [1, 768]
    CLS_feature_after_projection = outputs.text_embeds # shape -> [2, 512]; after normalization
    img_feature_after_projection = outputs.image_embeds # shape -> [1, 512]; after normalization
    loss = outputs.loss
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

def zero_shot_evaluation_huggingface(image_dataset_name, test_labels, ckpt = "openai/clip-vit-base-patch32"):
    # Load the model
    device = "cuda" if torch.cuda.is_available() else "cpu" 
    model = CLIPModel.from_pretrained(ckpt).to(device)
    #zero shot evaluation
    preprocess = Compose([
        Resize(size=224),
        CenterCrop(size=(224, 224)),
        ToTensor(),
        Normalize(mean=(0.48145466, 0.4578275, 0.40821073), 
                std=(0.26862954, 0.26130258, 0.27577711)),
        ])
    dataloader = get_image_dataloader(image_dataset_name, preprocess, train = False)
    evaluate_classification_huggingface(dataloader, test_labels, model, device)

# def play_with_data_loader(root_path, dir, batch_size, option = 'train'):
#     transform_dict = {
#         'train': transforms.Compose(
#         [transforms.RandomResizedCrop(224),
#          transforms.RandomHorizontalFlip(),
#          transforms.ToTensor(),
#          transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                               std=[0.229, 0.224, 0.225]),
#          ]),
#         'val': transforms.Compose(
#         [transforms.Resize(224),
#          transforms.ToTensor(),
#          transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                               std=[0.229, 0.224, 0.225]),
#          ])}
#     data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict[option])
#     data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
#     return data_loader 

if __name__ == '__main__':
    #play_with_tokenizer()
    # play_with_feature_extrator()
    # play_with_huggingface_clip()

    #cifar_labels = obtain_cifar_classes(root = '/nobackup/dataset_myf', which_cifar='CIFAR-10')
    #zero_shot_evaluation_huggingface(image_dataset_name = 'CIFAR-10', test_labels = cifar_labels)

    play_with_tokenizer_and_model(lang = 'en')