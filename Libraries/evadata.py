import torch
from skimage import io
import numpy as np
from torch.utils.data import Dataset
import random

def alldata(path):
    class_ids = [line.strip() for line in open( path + 'wnids.txt', 'r')]
    id_dict = {x:i for i, x in enumerate(class_ids)}
    all_classes = {line.split('\t')[0] : line.split('\t')[1].strip() for line in open( path + 'words.txt', 'r')}
    class_names = [all_classes[x] for x in class_ids]

    images = []
    labels = []

    # train data
    for value, key in enumerate(class_ids):
        images += [f'{path}train/{key}/images/{key}_{i}.JPEG' for i in range(500)]
        labels += [value for i in range(500)]

    # validation data
    for line in open( path + 'val/val_annotations.txt'):
        img_name, class_id = line.split('\t')[:2]
        images.append(f'{path}val/images/{img_name}')
        labels.append(id_dict[class_id])

    dataset = list(zip(images, labels))
    random.shuffle(dataset)

    return dataset, class_names

class TinyImagenetDataset(Dataset):
    """Tine Imagenet dataset reader."""

    def __init__(self, data, transform=None):
        """
        Args:
            data (string): zipped images and labels.
        """
        self.transform = transform
        self.images, self.labels = zip(*data)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.images[idx], as_gray=False, pilmode="RGB")

        if self.transform:
            image = self.transform(image)
        return image, self.labels[idx] #label

class parrotDataset(Dataset):

    def __init__(self, data, bgtr=None,fgbgtr=None,masktr=None,depthtr=None):
        
        self.bgtr = bgtr
        self.fgbgtr = fgbgtr 
        self.masktr = masktr
        self.depthtr = depthtr
        self.bg,self.fgbg,self.mask,self.depth = zip(*data)

    def __len__(self):
        return len(self.fgbg)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        bg = io.imread(self.bg[idx], as_gray=False, pilmode="RGB")
        fgbg = io.imread(self.fgbg[idx], as_gray=False, pilmode="RGB")
        mask = io.imread(self.mask[idx], as_gray=False, pilmode="RGB")
        depth = io.imread(self.depth[idx], as_gray=False, pilmode="RGB")

        if self.bgtr:
            bg_img = self.bgtr(bg)
        if self.fgbgtr:
            fgbg_img = self.fgbgtr(fgbg)
        if self.masktr:
            mask_img = self.masktr(mask)
        if self.depthtr:
            depth_img = self.depthtr(depth)

        return list([bg_img,fgbg_img,mask_img,depth_img])