# Project Structure (to be updated)
Out-of-distribution with language supervision
- Supported dataset: 'CIFAR-10', 'CIFAR-100', 'ImageNet', 'ImageNet10', 'ImageNet100', 'ImageNet-subset','ImageNet-dogs', 'bird200', 'car196','flower102','food101','pet37'

- `play_with_clip.py`: ID zero-shot classification and ID fine-tuning (with img encoder). Currently we have three options: 
   -  evaluate zero shot performance of CLIP: call `zero_shot_evaluation_CLIP(image_dataset_name, test_labels, ckpt)`
   -  fine-tune CLIP image encoder and test (linear probe): call `linear_probe_evaluation_CLIP(image_dataset_name)`
   -  play with SkImages: call `play_with_skimage()`


- `eval_ood_detection.py`: OOD detection for CIFAR-10, CIFAR-100, and ImageNet-1K as ID. Supported scores:
    -  'Maha', 'knn', 'analyze', # img encoder only; feature space 
    - 'energy', 'entropy', 'odin', # img->text encoder; feature space
    - 'MIP', 'MIPT','MIPT-wordnet', 'fingerprint', 'MIP_topk', # img->text encoder; feature space
    - 'MSP', 'energy_logits', 'odin_logits', # img encoder only; logit space
    - 'MIPCT', 'MIPCI', 'retrival', 'nouns' # text->img encoder; feature space

- `play_with_clip.ipynb`: contains various visualization methods for trained CLIP model.

- `captions.ipynb`: Notebook used to generated captions using the Oscar model from Microsoft. This assumes you have
cloned and installed [Oscar](https://github.com/microsoft/Oscar) and
[scene\_graph\_benchmark](https://github.com/microsoft/scene_graph_benchmark) in the directory running the notebook
from (you can change these directories in the notebook).

