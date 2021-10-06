from torchvision import transforms
import torch

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(mean=mean, std=std)
resnet_transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01),
    transforms.ToTensor(),
    normalize,
])

resnet_transform_val = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    normalize,
])

gan_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5),
])

random_affine = transforms.RandomAffine(
    degrees=(-20, 20),
    translate=(0.5,0.5),
    scale=(1.0, 1.0),
    # scale=(0.5, 2.0),
    # shear=(-5,5,-5,5),
    shear=None
)

gan_transform_aug = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(),
    random_affine,
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)])

def get_categories(filename):
    with open(filename, 'r') as f:
        categories = f.read().strip().split('\n')
    return categories

def get_labels(filename):
    with open(filename, 'r') as f:
        labels = f.read().strip().split('\n')
    lst = []
    for l in labels:
        tmp = [int(x) for x in l.split()]
        lst.append(tmp)
    return torch.tensor(lst).float()

def label2ingredients(label, categories):
    return '\n'.join([categories[i] for i in label.nonzero(as_tuple=False)])