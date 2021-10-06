import torch
from torch.nn import functional as F
from torchvision import transforms
import numpy as np
import os
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import lmdb
from io import BytesIO
from glob import glob
from pathlib import Path

import sys
sys.path.append('../')
import common
from datasets import utils
from datasets import true_label_50
from datasets.pizza3d import normalize_view_label

def _load_one_pizza(idx, env, transform):
    with env.begin(write=False) as txn:
        key = f'{idx}'.encode('utf-8')
        ingrs = txn.get(key).decode('utf-8')
        if not ingrs:
            ingrs = 'empty'
        txt = ingrs

        key = f'{256}-{idx}'.encode('utf-8')
        img_bytes = txn.get(key)
        
    buffer = BytesIO(img_bytes)
    img = Image.open(buffer)
    img = transform(img)
    return img, txt

 
class Pizza10DatasetFromImage(torch.utils.data.Dataset):
    def __init__(
        self,
        part='all',
        data_dir=f'{common.ROOT}/data/Pizza10/',
        transform=utils.resnet_transform_train,
    ):
        img_dir = f'{data_dir}/images'
        self.names = sorted(glob(f'{img_dir}/*.jpg'))
        self.transform = transform
        self.categories = utils.get_categories(os.path.join(data_dir, 'categories.txt'))
        self.labels = utils.get_labels(os.path.join(data_dir, 'imageLabels.txt'))

        # split by training and validation (8/2)
        split_point = int(0.8 * len(self.names))
        if part == 'train':
            self.names = self.names[:split_point]
            self.labels = self.labels[:split_point]
        elif part == 'val':
            self.names = self.names[split_point:]
            self.labels = self.labels[split_point:]
        
    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        filename = self.names[index]
        label = self.labels[index]
        img = Image.open(filename).convert('RGB')
        img = self.transform(img)
        tgt = {}
        tgt['raw_label'] = utils.label2ingredients(label, self.categories)
        tgt['ingr_label'] = label
        return img, tgt


class Pizza10Dataset(Dataset):
    def __init__(
       self,
       data_dir=f'{common.ROOT}/data/Pizza10/',
       part='all', 
       transform=utils.resnet_transform_train
    ):
        lmdb_file = os.path.join(data_dir, 'data.lmdb')
        self.categories = utils.get_categories(os.path.join(data_dir, 'categories.txt'))
        self.labels = utils.get_labels(os.path.join(data_dir, 'imageLabels.txt'))
        length = len(self.labels)

        self.env = lmdb.open(
            lmdb_file,
            max_readers=32,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        if not self.env:
            raise IOError('Cannot open lmdb dataset', lmdb_file)

        self.names = range(length)
        # split by training and validation (8/2)
        split_point = int(0.8 * length)
        if part == 'train':
            self.names = self.names[:split_point]
        elif part == 'val':
            self.names = self.names[split_point:]

        self.transform = transform
        self.img_dir = os.path.join(data_dir, 'images/')

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        # image
        idx = self.names[index]

        img, txt = _load_one_pizza(idx, self.env, self.transform)
        label = self.labels[idx]
        
        target = {}
        target['raw_label'] = txt
        target['ingr_label'] = label.float()
        return img, target


label2base = {
    0: 'whole_round',
    1: 'sliced_whole_round'
}

class LabeledPizza10Subset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir='../data/Pizza10/',
        transform=utils.resnet_transform_train,
    ):
        self.transform = transform
        self.categories = utils.get_categories(os.path.join(data_dir, 'categories.txt'))
        self.ingr_labels = utils.get_labels(os.path.join(data_dir, 'imageLabels.txt'))
        self.img_dir = Path(data_dir) / 'images'

    def __len__(self):
        return len(true_label_50.labels)

    def __getitem__(self, index):
        img_name = true_label_50.img_index_dict[index]

        img_path = os.path.join(self.img_dir, f'{img_name}.jpg')
        img = Image.open(img_path)
        img = self.transform(img)
        views_and_base = true_label_50.labels[img_name]
        tgt = {}
        tgt['ingr_label'] = self.ingr_labels[int(img_name)-1]
        view_label = torch.tensor([
            true_label_50.angle(views_and_base), 
            true_label_50.scale(views_and_base), 
            true_label_50.x_axis(views_and_base), 
            true_label_50.y_axis(views_and_base)
        ]).float()
        tgt['view_label'] = normalize_view_label(view_label)
        
        tgt['base_label'] = torch.tensor(views_and_base['base']).float()
        tgt['raw_label'] = utils.label2ingredients(tgt['ingr_label'], self.categories)
        tgt['raw_label'] += '\n'
        tgt['raw_label'] += '\n'.join([f'{x:.1f}' for x in view_label])
        tgt['raw_label'] += '\n'
        tgt['raw_label'] += label2base[int(tgt['base_label'])]

        return img, tgt


if __name__ == '__main__':
    from tqdm import tqdm
    bs = 64
    nrow = int(np.sqrt(64))
    os.makedirs('ignore/', exist_ok=True)
    torch.manual_seed(8)

    dataset = Pizza10Dataset(data_dir=f'{common.ROOT}/data/Pizza10/', part='train', transform=utils.resnet_transform_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)
    print(len(dataset), len(dataloader))
    for img, tgt in tqdm(dataloader):
        print(img.shape)
        # print(tgt['raw_label'])
        # print(tgt['ingr_label'])
        common.save_captioned_image(tgt['raw_label'], img, 'ignore/pizza10_batch.jpg', font=10, nrow=nrow)
        break

    dataset = LabeledPizza10Subset(data_dir=f'{common.ROOT}/data/Pizza10/', transform=utils.resnet_transform_train)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=False)
    print(len(dataset), len(dataloader))
    for img, tgt in tqdm(dataloader):
        print(img.shape)
        # print(tgt['raw_label'])
        # print(tgt['ingr_label'])
        common.save_captioned_image(tgt['raw_label'], img, 'ignore/labeledPizza10Subset_batch.jpg', font=10, nrow=nrow)
        break