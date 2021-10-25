import streamlit as st
import torch
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import uuid

import sys
sys.path.append('../')
from common import requires_grad, ROOT
from common import normalize, resize
from mpg.train import load_mpg
from ingr_classifier.train import load_classifier
from view_regressor.train import load_regressor
from streamlit.report_thread import get_report_ctx
ctx = get_report_ctx()
session_id = ctx.session_id

from datasets.pizza3d import view_attr, view_ranges, normalize_view_label
default_view_label = [0.0, 2.0, 0.0, 0.0]
view_ranges = view_ranges.tolist()

@st.experimental_memo(show_spinner=False)
def load_models(ckpt_path, classifier_path, regressor_path, device):

    _, _, label_encoder, _, _, g_ema, _, _, _ = load_mpg(ckpt_path, device=device)
    requires_grad(label_encoder, False)
    requires_grad(g_ema, False)
    label_encoder.eval()
    g_ema.eval()

    _, _, classifier, _ = load_classifier(classifier_path, device=device)
    requires_grad(classifier, False)
    classifier.eval()

    _, _, regressor, _ = load_regressor(regressor_path, device=device)
    requires_grad(regressor, False)
    regressor.eval()

    return label_encoder, g_ema, classifier, regressor

@st.experimental_memo()
def load_categories(filename):
    with open(filename, 'r') as f:
        categories = f.read().strip().split('\n')
    return categories

def compute_mean_latent(g_ema, truncation, truncation_mean):
    if truncation < 1:
        with torch.no_grad():
            mean_latent = g_ema.mean_latent(truncation_mean)
    else:
        mean_latent = None
    return mean_latent

def to_numpy_img(tensor):
    npimg = tensor.detach().cpu().numpy()
    npimg = (npimg-npimg.min()) / (npimg.max()-npimg.min())
    return np.transpose(npimg, (1,2,0))

@torch.no_grad()
def generate_cs(noise, label, label_encoder, g_ema, truncation, mean_latent):
    text_outputs = label_encoder(label)
    sample, _ = g_ema(
        [noise], text_outputs, truncation=truncation, truncation_latent=mean_latent, randomize_noise=False
    )
    return sample

# @st.cache()
def generate_prediction_figure(img, classifier, categories):
    fake_img = normalize(img)
    fake_img = resize(fake_img, size=224)
    output = classifier(fake_img)
    probs = torch.sigmoid(output).squeeze().cpu().numpy()
    # gt = binary_label.detach().cpu().squeeze().numpy()
    fig, ax = plt.subplots()
    ax.title.set_text('Ingredients Prediction')
    ind = np.arange(len(probs))
    width = 0.3
    ax.barh(ind, probs, width, color='green', label='Prediction')
    # ax.barh(ind + width, gt, width, color='red', label='Ground Truth')
    ax.set(yticks=ind + width, yticklabels=categories)
    ax.legend()
    ax.set_xlim(0.0, 1.0)
    return fig

def config_input(col, col_name, categories, generator, device):
    col.header('Ingredients')
    ingr = 'Pepperoni' if col_name=='left' else 'Fresh basil'
    ingr_label = torch.zeros(1, 10).to(device)
    for i, category in enumerate(categories):
        val = col.checkbox(category, value=True if category==ingr else False, key=f'{col_name}-{category}')
        ingr_label[:,i] = float(val)

    view_label = torch.zeros(1, 4).to(device)
    for i in range(4):
        view_label_key = f'{col_name}_view_label_{i+1}'
        if view_label_key not in st.session_state:
            value = default_view_label[i]
            st.session_state[view_label_key] = value
        val = col.slider(view_attr[i], min_value=view_ranges[i][0], max_value=view_ranges[i][1], value=st.session_state[view_label_key], step=0.01, key=view_label_key)
        view_label[:,i] = val
    view_label = normalize_view_label(view_label)

    label = torch.cat([ingr_label, view_label],dim=1)
    
    truncation = col.number_input('Diversity Level', min_value=0.0, max_value=1.0, value=0.7, step=0.1, key=f'{col_name}-diversity')
    truncation_mean = 4096
    mean_latent = compute_mean_latent(generator, truncation, truncation_mean)

    noise = torch.zeros(1, 256).to(device)
    if col.button('Refresh', key=f'{col_name}-refresh'):
        for i in range(256):
            noise_key = f'{col_name}_noise_{i+1}'
            del st.session_state[noise_key]
        
    with col.expander('Styles'):
        for i in range(256):
            noise_key = f'{col_name}_noise_{i+1}'
            if noise_key not in st.session_state:
                value = np.random.randn()
                st.session_state[noise_key] = value
            val = st.slider(f'Dimension{i+1}', min_value=-10.0, max_value=10.0, value=st.session_state[noise_key], step=0.01, key=noise_key)
            noise[:,i] = val
    
    return label, noise, truncation, mean_latent

def main():
    st.set_page_config(page_title='MPG2', layout="wide")
    st.title('MPG2 - Multi-attribute Pizza Generator: Cross-domain Attribute Control with Conditional StyleGAN')
    txt = '''
    | [Back to FoodAI](http://foodai.cs.rutgers.edu) | [Paper](https://arxiv.org/abs/2110.11830) | [Code](https://github.com/klory/MPG2) |
    |---|---|---|

    The image in the middle is the interpolation between the two images.\
        Feel free to change the ingredients, **VIEW ATTRIBUTES**, style noise, diversity level and interpolation points.
    '''
    st.markdown(txt)

    categories_filename = f'{ROOT}/data/Pizza10/categories.txt'
    categories = load_categories(categories_filename)

    ckpt_path = f'{ROOT}/mpg/runs/30cupu9m/00260000.ckpt'
    classifier_path = f'{ROOT}/ingr_classifier/runs/pizza10/1t5xrvwx/batch5000.ckpt'
    regressor_path = f'{ROOT}/view_regressor/runs/pizza3d/1ab8hru7/00004999.ckpt'
    device='cuda'
    label_encoder, generator, classifier, regressor = load_models(ckpt_path, classifier_path, regressor_path, device)

    left_bar, left_col, mid_col, right_col, right_bar = st.columns([1,1,1,1,1])

    label_1, noise_1, truncation_1, mean_latent_1 = config_input(left_bar, 'left', categories, generator, device)
    label_2, noise_2, truncation_2, mean_latent_2 = config_input(right_bar, 'right', categories, generator, device)


    with st.spinner('Generating new images...'):
        sample = generate_cs(noise_1, label_1, label_encoder, generator, truncation_1, mean_latent_1)
        img_np = to_numpy_img(sample.squeeze(0))
        # left_col.header("Image A")
        left_col.image(img_np, use_column_width=True)
        fig = generate_prediction_figure(sample, classifier, categories)
        left_col.pyplot(fig)

        sample = generate_cs(noise_2, label_2, label_encoder, generator, truncation_2, mean_latent_2)
        img_np = to_numpy_img(sample.squeeze(0))
        # right_col.header("Image B")
        right_col.image(img_np, use_column_width=True)
        fig = generate_prediction_figure(sample, classifier, categories)
        right_col.pyplot(fig)

        mid_col.header("Interpolation")
        ingr_from_src2 = mid_col.slider('Ingr+View from Image B', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        noise_from_src2 = mid_col.slider('Style from Image B', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
        label = ingr_from_src2*label_2 + (1-ingr_from_src2)*label_1
        noise = noise_from_src2*noise_2 + (1-noise_from_src2)*noise_1
        sample = generate_cs(noise, label, label_encoder, generator, truncation_1, mean_latent_1)
        image_np = sample[0].cpu().numpy().transpose(1,2,0)
        # image_np = np.uint8( image_np * 255.0 )
        image_np = np.ascontiguousarray(image_np, dtype=np.uint8)
        img_np = to_numpy_img(sample.squeeze(0))
        mid_col.image(img_np, use_column_width=True)
        fig = generate_prediction_figure(sample, classifier, categories)
        mid_col.pyplot(fig)
        
        # uuid_str = uuid.uuid4().hex
        # img_pil = Image.fromarray(image_np)
        # img_pil.save(f"{ROOT}/mpg_server/static/generated_images/{uuid_str}.jpeg")
        # plt.savefig(f"{ROOT}/mpg_server/static/generated_images/{uuid_str}_chart.jpeg")

        
if __name__ == "__main__":
    main()
