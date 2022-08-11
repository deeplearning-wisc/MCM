# Delving into OOD Detection with Vision-Language Representations

Recognizing out-of-distribution (OOD) samples is critical for machine learning systems deployed in the open world. The vast majority of OOD detection methods are driven by a single modality (e.g., either vision or language), leaving the rich information in multi-modal representations untapped. Inspired by the recent success of vision-language pre-training, this paper enriches the landscape of OOD detection from a single-modal to a multi-modal regime. Particularly, we propose Maximum Concept Matching (MCM), a simple yet effective zero-shot OOD detection method based on aligning visual features with textual concepts. We contribute in-depth analysis and theoretical insights to understand the effectiveness of MCM. Extensive experiments demonstrate that our proposed MCM achieves superior performance on a wide variety of real-world tasks. MCM with vision-language features outperforms a common baseline with pure visual features on a hard OOD task with semantically similar classes by 56.60% (FPR95).

# Links

ArXiv

# Environment Setup

```sh
conda create -n clip-ood python=3.7 -y
conda activate clip-ood

# Install GPU version of pytorch, please verify your own CUDA toolkit version
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge

# Install dependencies
pip install -r requirments.txt
```

# Data Preparation

For complete information, refer to Appendix B.3 of the paper. The default dataset location is `./datasets/`, which can be changed in `settings.yaml`.

## In-distribution Datasets

- [`CUB-200`](http://www.vision.caltech.edu/datasets/cub_200_2011/), [`Standford-Cars`](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [`Food-101`](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), [`Oxford-Pet`](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [`ImageNet`](https://image-net.org/challenges/LSVRC/2012/index.php#), [`ImageNet-10`](https://github.com/alvinmingwisc/CLIP_OOD/blob/clean-up/dataloaders/ImageNet-10-classlist.csv), [`ImageNet-20`](https://github.com/alvinmingwisc/CLIP_OOD/blob/clean-up/dataloaders/ImageNet-20-classlist.csv)

Please download the full ImageNet dataset from the link; the other datasets can be automatically downloaded as the experiments run.

## Out-of-Distribution Datasets

- [iNaturalist](https://arxiv.org/abs/1707.06642), [SUN](https://vision.princeton.edu/projects/2010/SUN/), [Places](https://arxiv.org/abs/1610.02055), [Texture](https://arxiv.org/abs/1311.3618)

We use the large scale OOD datasets curated by [Huang et al. 2021](https://arxiv.org/abs/2105.01879). Please follow instruction from the this [repository](https://github.com/deeplearning-wisc/large_scale_ood#out-of-distribution-dataset) to download the cleaned datasets, where overlaps with ImageNet are removed.

The overall file structure:

```
CLIP_OOD
|-- datasets
    |-- ImageNet
    |-- CUB-200
    |-- Food-101
    |-- iNaturalist
    ...
```

# Experiments

## OOD Detection

The main entry point for running OOD detection experiments is `eval_ood_detection.py`. Here are the list of arguments:

- `--name`: A unique ID for the experiment, can be any string.
- `--seed`: Random seed for the experiments. (We used 4.)
- `--gpu`: The indexes of the GPUs to use. For example `--gpu=0 1 2`.
- `--in_dataset`: The in-distribution dataset.
  - Accepts: `CIFAR-10`, `CIFAR-100`, `ImageNet`, `ImageNet10`, `ImageNet20`, `ImageNet100`, `bird200`, `car196`, `flower102`, `food101` , `pet37`,
  <!-- - `--out_datasets`: The out-of-distribution datasets, we accept multiple ones.
  - Accepts: `iNat`, `SUN`, `Places`, `DTD`, `ImageNet10`, `ImageNet20` -->
- `-b`, `--batch_size`: Mini-batch size; 1 for nouns score; 75 for odin_logits; 512 for other scores [clip].
- `--epoch`: Number of epochs to run if doing linear probe.
- `--model`: The model architecture to extract features with.
  - Accepts: `CLIP`, `CLIP-Linear`, `ViT`, `ViT-Linear`. (`-Linear` is the linear probe version of the model.)
- `--CLIP_variant`: Specifies the pretrained CLIP encoder to use.
  - Accepts: `ViT-B/32`, `ViT-B/16`, `RN50x4`, `ViT-L/14`.
- `--classifier_ckpt`: Specifies the linear probe classifier to load.
- `--score`: The OOD detection score, we accept any of the following:

  - `MCM`: Maximum Concept Matching, Our main result; Correspond to Table 1, 2 in our paper.
  - `Maha`: [Mahalanobis score](https://arxiv.org/abs/1807.03888), Correspond to figure 5 in the paper. First time running wil generate class-wise means and precision matrices used in calculation.
  - `energy`: [Energy based score](https://proceedings.neurips.cc/paper/2020/hash/f5496252609c43eb8a3d147ab9b9c006-Abstract.html), Correspond to Table 6 in our paper.
  - `max-logit`: Cosine similarity without softmax.
  - `entropy`, `var`, `scaled`: Respectively: ngative entropy of softmax scaled cosine similarities, variance of cosine similarities, and the scaled difference between the largest and second largest cosine similarities. Correspond to Table 7 in our paper.
  - `MSP`: [Maximum Softmax Probability](https://arxiv.org/abs/1610.02136); Classic baseline score.

The results are stored in the folder `./results/`. The format is a csv.

## Fine-tuning

[TODO]

# Reproduction

Here are the commands to reproduce numerical results of our paper, note that we ran our experiments on a single GTX 2080 GPU.

## Table 1

```sh
eval_ood_detection.py \
    --in_dataset={ImageNet10, ImageNet20, ImageNet100, bird200, car196, flower102, food101/pet37} \
    --out_dataset=iNat SUN Places DTD
    --model=CLIP --CLIP_variant=ViT-B/16 \
    --score=MCM \
    --batch_size=512
```

## Table 2

```sh
# zero shot
eval_ood_detection.py \
    --in_dataset=ImageNet --model=CLIP --CLIP_variant={ViT-B/16, ViT-L/14}
    --score=MCM
    --batch_size=512

# Fort et al, MSP
eval_ood_detection.py \
    --in_dataset=ImageNet --model=ViT --CLIP_variant={ViT-B/16, ViT-L/14}
    --score={Maha, MSP}
    --batch_size=512
```

## Table 3

```
eval_ood_detection.py \
    --in_dataset={ImageNet-10, ImageNet-20, Waterbirds}
    --out_dataset={ImageNet-20, ImageNet-10, Waterbirds-Spurious-OOD}
    --model=CLIP --CLIP_variant=ViT-B/16
    --score={MSP, Maha, MCM}
```
