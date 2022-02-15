# Project Structure
Out-of-distribution with language supervision

`play_with_clip.py` is the main file. Currently we have three options: 

1) evaluate zero shot performance of CLIP: call `zero_shot_evaluation_CLIP(image_dataset_name, test_labels, ckpt)`

2) fine-tune CLIP image encoder and test (linear probe): call `linear_probe_evaluation_CLIP(image_dataset_name)`

3) play with SkImages: call `play_with_skimage()`

# Week2 Record 
New dataset from ImageNet: 
- 10 Classes from ImageNet-1k
    - location: inst-01 /nobackup/ImageNet
    - classes: n04552348 (plane), n04285008 (car/automobile), n01530575 (bird), n02123597 (cat), n02422699 (antelope), n02107574 (dog) ,n01641577 (frog)
       , n01728572 (snake), n03095699 (ship), n03417042 (truck)
- Tasks:
  - generate Captions
  -
# Week1 Record 

Q: What are desirable properties of pre-trained CLIP?

- It recognizes objects (not background)! -> it's been shown that CLIP is robust across background shift
- It associates image representations with label descriptions
- If the true label is available, it assigns high confidence

Q: Problems of pre-trained CLIP?

- If the true labels are not there, it can still be overconfident

Q: [detection side] Now we have text embedding, how to design a better detection score?

- feature-based approach:
    - NIPS-2021 cheating approach (assume OOD labels are known)
    - Inner product based
        - only using ID labels
        - find a fixed sef of template labels?
    - KNN
    - Mahalanobis score
    - improved Mahalanobis score
        - challenge: there is a mismatch between text and img feature spaces, although they have the same dimension
- logit-based approach: (to check if text embeddings are useful)
    - based on pre-trained CLIP - > linear probe
    - based on pre-trained ViT (without pre-training with text encoder) -> linear probe

Q: [fine-tuning side] Now we have an ID dataset, how to make CLIP aware of ID and OOD?

- baseline #1: just train with contrastive loss with ID only
- baseline #2: add K+1 class as "OOD" and use an auxiliary pool
    - challenge: CLIP is good at recognizing concrete objects; is it still good for abstract notions such as "OOD"?
- baseline #3: typical OE approach: use uniform distribution as proxy for outliers
- baseline #4: [grouping-based] similar to the MOS paper (say we have 8 big categories from ImageNet): some of them are ID (e.g. plants) and some of them are OOD (e.g. furniture). This is more abstract but more specific than “placeholder”. And during fine-tuning, we can assign ID and auxiliary outliers to those categories.
