import os
import json
import numpy as np
import copy
import json
import argparse
from torchvision import transforms
from PIL import Image
import math
import cv2
import torch
from torch.nn import functional as F
import pathlib

ROOT = pathlib.Path(__file__).parent.resolve()

def normalize(img):
    """
    normalize a batch
    """
    img = (img-img.min())/(img.max()-img.min())
    means = [0.485, 0.456, 0.406]
    stds = [0.229, 0.224, 0.225]
    for i in range(3):
        img[:,i] = (img[:,i]-means[i])/stds[i]
    return img

def resize(img, size=224):
    """
    resize a batch
    """
    return F.interpolate(img, size=(size, size), mode='bilinear', align_corners=False)

def process_img_for_classifier(img,size=224):
    return normalize(resize(img))

def load_categories(filename):
    with open(filename, 'r') as f:
        categories = f.read().strip().split('\n')
    return categories


def rotate_image(image, angle):
    """
    Rotates an OpenCV 2 / NumPy image around it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=cv2.INTER_LINEAR
    )

    return result


def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )


def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def clean_state_dict(state_dict):
    # create new OrderedDict that does not contain `module.`
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k[:min(6,len(k))] == 'module' else k # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def infinite_loader(loader):
    """
    arguments:
        loader: torch.utils.data.DataLoader
    return:
        one batch of data
    usage:
        data = next(infinite_loader(loader))
    """
    while True:
        for batch in loader:
            yield batch

def make_captioned_image(caption, image, font=20, color=(255,255,0), loc=(0,0), nrow=8, tint_color=(0,0,0), opacity=0.4, pad_value=0):
    import torch
    from PIL import Image, ImageFont, ImageDraw
    from torchvision.utils import make_grid
    assert len(caption) == len(image)
    font = ImageFont.truetype('UbuntuMono-R.ttf', font)
    imgs = image.cpu().numpy()
    imgs = (imgs-imgs.min()) / (imgs.max()-imgs.min())
    txted_imgs = []
    # how to draw transparent rectangle: https://stackoverflow.com/a/43620169/6888630
    opacity = int(255*opacity)
    for txt, img in zip(caption, imgs):
        img = img.transpose(1,2,0)
        img = Image.fromarray(np.uint8(img*255))
        
        # draw background
        img = img.convert("RGBA")
        x=y=0
        w, h = font.getsize(txt)
        h *= (len(txt.split('\n'))+1)
        overlay = Image.new('RGBA', img.size, tint_color+(0,))
        draw = ImageDraw.Draw(overlay)  # Create a context for drawing things on it.
        draw.rectangle(((x,y,x+w,y+h)), fill=tint_color+(opacity,))

        # Alpha composite these two images together to obtain the desired result.
        img = Image.alpha_composite(img, overlay)
        img = img.convert("RGB") # Remove alpha for saving in jpg format.
        
        # draw text
        draw = ImageDraw.Draw(img)
        draw.text(loc, txt, fill=color, font=font)

        img = transforms.ToTensor()(img)
        txted_imgs.append(img)
    txted_imgs = torch.stack(txted_imgs)
    big_pic = make_grid(txted_imgs, nrow=nrow, padding=2, normalize=True, pad_value=pad_value, scale_each=True)
    return big_pic

def save_captioned_image(caption, image, fp, font=20, color=(255,255,0), opacity=0.5, loc=(0,0), nrow=8, pad_value=0):
    grid = make_captioned_image(caption, image, opacity=opacity, font=font, color=color, loc=loc, nrow=nrow, pad_value=pad_value)
    ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(fp)

def load_recipes(file_path, part='', food_type=''):
    print(f'load recipes from {file_path}')
    with open(file_path, 'r') as f:
        info = json.load(f)
    if part:
        info = [x for x in info if x['partition']==part]
    if food_type:
        info = [x for x in info if food_type in x['title'].lower()]
    return info


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def dspath(ext, ROOT, **kwargs):
    return os.path.join(ROOT,ext)

class Layer(object):
    L1 = 'layer1'
    L2 = 'layer2'
    L3 = 'layer3'
    INGRS = 'det_ingrs'

    @staticmethod
    def load(name, ROOT, **kwargs):
        with open(dspath(name + '.json',ROOT, **kwargs)) as f_layer:
            return json.load(f_layer)

    @staticmethod
    def merge(layers, ROOT,copy_base=False, **kwargs):
        layers = [l if isinstance(l, list) else Layer.load(l, ROOT, **kwargs) for l in layers]
        base = copy.deepcopy(layers[0]) if copy_base else layers[0]
        entries_by_id = {entry['id']: entry for entry in base}
        for layer in layers[1:]:
            for entry in layer:
                base_entry = entries_by_id.get(entry['id'])
                if not base_entry:
                    continue
                base_entry.update(entry)
        return base


count_parameters = lambda model: sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    data_dir = '../data'
    print('load recipes (20 seconds)')
    recipes_original = Layer.merge(
        [Layer.L1, Layer.L2, Layer.INGRS], 
        os.path.join(data_dir, 'texts'))

    for rcp in recipes_original:
        rcp['instructions'] = [x['text'] for x in rcp['instructions']]
        rcp['ingredients'] = [x['text'] for x in rcp['ingredients']]

    with open(os.path.join(data_dir, 'original.json'), 'w') as f:
        json.dump(recipes_original, f, indent=2)

    with open(os.path.join(data_dir, 'original_withImage.json'), 'w') as f:
        recipes_original_with_image = [r for r in recipes_original if ('images' in r) and len(r['images'])>0]
        json.dump(recipes_original_with_image, f, indent=2)