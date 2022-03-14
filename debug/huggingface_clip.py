import torch
import torchvision.transforms as transforms
from torchvision import datasets
from PIL import Image
import requests
import numpy as np

def play_with_image_processing(url =  "http://images.cocodataset.org/val2017/000000039769.jpg"): 
    image = Image.open(requests.get(url, stream=True).raw) #PIL (pillow) image;
    np_image = np.array(image) #shape -> (480, 640, 3); PIL or Numpy array represents image with channels in the last dimension
    transform = transforms.ToTensor()
    tensor = transform(np_image) # shape -> torch.Size([3, 480, 640])
    return image

def play_with_huggingface_clip():
    from transformers import CLIPProcessor, CLIPModel
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    for name, parameter in model.named_parameters(): 
        print(name, parameter.requires_grad)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    image = play_with_image_processing()
    inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)
    # inputs.data.keys() -> dict_keys(['input_ids', 'attention_mask', 'pixel_values'])
    # inputs.data['pixel_values'].shape -> torch.Size([1, 3, 224, 224])
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image # this is the image-text similarity score
    probs = logits_per_image.softmax(dim=1) # we can take the softmax to get the label probabilities

def play_with_tokenizer():
    from transformers import CLIPTokenizerFast
    tokenizer = CLIPTokenizerFast.from_pretrained("openai/clip-vit-base-patch32")
    text_inputs = tokenizer("Hello world")
    # text_inputs.keys() -> dict_keys(['input_ids', 'attention_mask'])
    print("")


def play_with_data_loader(root_path, dir, batch_size, option = 'train'):
    transform_dict = {
        'train': transforms.Compose(
        [transforms.RandomResizedCrop(224),
         transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ]),
        'val': transforms.Compose(
        [transforms.Resize(224),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225]),
         ])}
    data = datasets.ImageFolder(root=root_path + dir, transform=transform_dict[option])
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    return data_loader 

if __name__ == '__main__':
    #play_with_tokenizer()
    play_with_huggingface_clip()