import torch
from torchvision import transforms
from PIL import Image
from torchvision.transforms import functional as TF
import os
from glob import glob
import numpy as np
import cv2

import sys
sys.path.append('../')
from common import save_captioned_image, ROOT
from datasets import utils

view_attr = ['Angle', 'Scale', 'Dx', 'Dy']

view_ranges = torch.tensor([
    [0.0, 75.0],
    [1.0, 3.0],
    [-112, 112],
    [-112, 112]
])

def normalize_angle(val):
    return val / 75.0

def normalize_scale(val):
    return val / 3.0

def normalize_dx(val):
    return val / 112.0

def normalize_dy(val):
    return val / 112.0

def normalize_view_label(val):
    if val.ndim == 1:
        return torch.tensor([normalize_angle(val[0]), normalize_scale(val[1]), normalize_dx(val[2]), normalize_dy(val[3])]).to(val.device)
    elif val.ndim == 2:
        return torch.stack([normalize_angle(val[:, 0]), normalize_scale(val[:, 1]), normalize_dx(val[:, 2]), normalize_dy(val[:, 3])], dim=1)

def denormalize_angle(val):
    return val * 75.0

def denormalize_scale(val):
    return val * 3.0

def denormalize_dx(val):
    return val * 112.0

def denormalize_dy(val):
    return val * 112.0

def denormalize_view_label(val):
    if val.ndim == 1:
        return torch.tensor([denormalize_angle(val[0]), denormalize_scale(val[1]), denormalize_dx(val[2]), denormalize_dy(val[3])]).to(val.device)
    elif val.ndim == 2:
        return torch.stack([denormalize_angle(val[:, 0]), denormalize_scale(val[:, 1]), denormalize_dx(val[:, 2]), denormalize_dy(val[:, 3])], dim=1)


class Pizza3DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir=f'{ROOT}/data/Pizza3D',
        size=224,
        transform=utils.resnet_transform_train,
    ):
        self.img_filenames = sorted(glob(f'{data_dir}/images/*.jpg'))
        self.size = size
        self.transform = transform
        self.categories = utils.get_categories(os.path.join(data_dir, 'categories10.txt'))
        self.labels = utils.get_labels(os.path.join(data_dir, 'labels10.txt'))

    def __len__(self):
        return len(self.img_filenames)
    
    def __getitem__(self, index):
        filename = self.img_filenames[index]
        img = Image.open(filename)
        basename = os.path.basename(filename)
        cg_model_idx, angle, _ = basename.split('_')
        cg_model_idx = int(cg_model_idx)
        angle = float(angle)

        img = TF.resize(img, (self.size, self.size))
        
        rec = transforms.RandomAffine.get_params(
            degrees=(-20, 20),
            translate=(0.5,0.5),
            scale_ranges=(0.5,2),
            # shears=(-5,5,-5,5),
            shears=None,
            img_size=(self.size,self.size),
        )
        
        img = TF.affine(img, *rec)
        
        # inteligently background
        img = np.array(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY_INV)
        img = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
        img = Image.fromarray(img)

        img = self.transform(img)
        # rotation = rec[0]
        dx = rec[1][0]
        dy = rec[1][1]
        scale = rec[2]
        # shear_x = rec[3][0]
        # shear_y = rec[3][1]
        raw_view_label = torch.FloatTensor([angle, scale, dx, dy])
        view_label = normalize_view_label(raw_view_label)
        ingr_label = self.labels[cg_model_idx-1]
        raw_label = utils.label2ingredients(ingr_label, self.categories)
        for label in raw_view_label:
            raw_label += f'\n{label:.2f}'

        tgt = {}
        tgt['raw_label'] = raw_label
        tgt['label'] = torch.cat([ingr_label, view_label])
        tgt['ingr_label']=ingr_label
        tgt['view_label']=view_label
        return img, tgt


if __name__ == '__main__':
    dataset = Pizza3DDataset()
    bs = 64
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, num_workers=8, shuffle=False)
    print(len(dataset), len(dataloader))
    os.makedirs('ignore/', exist_ok=True)
    for img, tgt in dataloader:
        print(img.shape)
        print(tgt['label'].shape)
        caption = []
        save_captioned_image(tgt['raw_label'], img, 'ignore/pizza3D-batch.jpg', font=10, pad_value=1)
        break