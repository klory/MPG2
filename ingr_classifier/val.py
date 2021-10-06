import torch
import numpy as np
from torch import nn
import os
import sys
from tqdm import tqdm
from torchnet import meter
import pdb
sys.path.append('../')
from ingr_classifier.args import get_parser
import common 
from ingr_classifier.train import load_dataset, load_classifier

args = get_parser().parse_args()

# ****** For Reproductability *******
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.benchmark = True
# torch.backends.cudnn.deterministic = True
device = 'cuda'

# ****** model *******
if not args.ckpt_path:
    args.ckpt_path = common.ROOT / 'ingr_classifier/runs/pizza10/1t5xrvwx/batch5000.ckpt'
ckpt_args, _, model, _ = load_classifier(args.ckpt_path) 
# print(ckpt_args)
model.eval()
model = model.to(device)
model = nn.DataParallel(model)

# ****** dataset *******
_, val_set, categories = load_dataset(args)
print(categories)
args.categories = categories
args.num_categories = len(categories)
print(len(val_set))
val_loader = torch.utils.data.DataLoader(val_set, batch_size=2*args.batch_size, shuffle=False)
print(len(val_loader))


running_output = []
running_label = []
pbar = tqdm(val_loader)
with torch.no_grad():
    for img, tgt in pbar:
        label = tgt['ingr_label']
        img, label = img.to(device), label.to(device)
        bs = img.shape[0]
        output = model(img)
        running_output.append(output)
        running_label.append(label)

running_output = torch.sigmoid(torch.cat(running_output, dim=0))
running_label = torch.cat(running_label, dim=0)
mtr = meter.APMeter()
mtr.add(running_output, running_label)
# call mtr.value() will return the APs (average precision) for each \
# class, so the output will have shape [K]
APs = mtr.value()
mAP = APs.mean() # mean average precision
print(f'mAP = {mAP:.4f}')

# compute confusion matrix
from sklearn.metrics import multilabel_confusion_matrix
y_true = running_label.cpu().numpy()
y_pred = (running_output>0.5).cpu().numpy()
print(y_true.shape, y_pred.shape)
mats = multilabel_confusion_matrix(y_true, y_pred)
for cat, mat, ap in zip(args.categories, mats, APs):
    print(cat)
    print(ap)
    print(mat)
    print()
