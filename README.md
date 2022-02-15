# CLIP_OOD
Out-of-distribution with language supervision

Currently we have three options: 

1) evaluate zero shot performance of CLIP: call `zero_shot_evaluation_CLIP(image_dataset_name, test_labels, ckpt)`

2) fine-tune CLIP image encoder and test (linear probe): call `linear_probe_evaluation_CLIP(image_dataset_name)`

3) play with SkImages: call `play_with_skimage()`
