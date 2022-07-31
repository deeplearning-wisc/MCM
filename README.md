# Delving into OOD Detection with Vision-Language Representations

Recognizing out-of-distribution (OOD) samples is critical for machine learning systems deployed in the open world. The vast majority of OOD detection methods are driven by a single modality (e.g., either vision or language), leaving the rich information in multi-modal representations untapped. Inspired by the recent success of vision-language pre-training, this paper enriches the landscape of OOD detection from a single-modal to a multi-modal regime. Particularly, we propose Maximum Concept Matching (MCM), a simple yet effective zero-shot OOD detection method based on aligning visual features with textual concepts. We contribute in-depth analysis and theoretical insights to understand the effectiveness of MCM. Extensive experiments demonstrate that our proposed MCM achieves superior performance on a wide variety of real-world tasks. MCM with vision-language features outperforms a common baseline with pure visual features on a hard OOD task with semantically similar classes by 56.60% (FPR95).

# Links

ArXiv

# Environment Setup

# Data Preparation

For complete information, refer to Appendix B.3 of the paper.

## In-distribution Datasets

- [`CUB-200`](http://www.vision.caltech.edu/datasets/cub_200_2011/), [`Standford-Cars`](http://ai.stanford.edu/~jkrause/cars/car_dataset.html), [`Food-101`](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/), [`Oxford-Pet`](https://www.robots.ox.ac.uk/~vgg/data/pets/)
- [`ImageNet`](https://image-net.org/challenges/LSVRC/2012/index.php#), `ImageNet-10`, `ImageNet-20`

Please download ImageNet from the link; the other datasets can be automatically downloaded as the experiments run. The default dataset location is `./datasets/`, which can be changed in `settings.yaml`. The overall file structure:

```
CLIP_OOD
|-- datasets
    |-- ImageNet
    |-- ImageNet-10
        |-- classlist.csv
    |-- ImageNet-20
        |-- classlist.csv
```

# Experiments
