import os, json


def obtain_ImageNet_subset_classes(loc):
    # sort by values
    with open(os.path.join(loc, 'class_list.txt')) as f:
        class_set = [line.strip() for line in f.readlines()]

    class_name_set = []
    with open('data/ImageNet/imagenet_class_index.json') as file: 
        class_index_raw = json.load(file)
        class_index = {cid: class_name for cid, class_name in class_index_raw.values()}
        class_name_set = [class_index[c] for c in class_set]
    class_name_set = [x.replace('_', ' ') for x in class_name_set]

    return class_name_set



if __name__ == "__main__":
    name = 'test_imagenet100_10'
    seed  = 4
    loc = os.path.join('./data', f'ImageNet100', f'{name}_seed_{seed}')
    class_list = obtain_ImageNet_subset_classes(loc)
    print(class_list)