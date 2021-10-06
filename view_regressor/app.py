# CUDA_VISIBLE_DEVICES=0 streamlit run val_on_pizza10.py

import streamlit as st
from PIL import Image
from pathlib import Path
import torch
from matplotlib import pyplot as plt
from tqdm import tqdm
import pdb
import numpy as np
from scipy import stats
import os
from streamlit_plotly_events import plotly_events

import sys
sys.path.append('..')
import common
from view_regressor.train import load_regressor
from datasets.pizza3d import Pizza3DDataset, view_ranges
from datasets.pizza10 import Pizza10Dataset, LabeledPizza10Subset
from datasets.pizza3d import denormalize_view_label
from datasets.utils import resnet_transform_train, resnet_transform_val
from common import ROOT

device = 'cuda'
np.random.seed(8)
torch.manual_seed(8)

def load_models(ckpt_path='view_regressor/runs/pizza3d/1ab8hru7/00004999.ckpt', device='cuda'):
    _, _, model, _ = load_regressor(ckpt_path)
    model = model.eval().to(device)
    common.requires_grad(model, False)
    return model

class Attr:
    names = ['angle', 'scale', 'dx', 'dy']
    ranges = view_ranges

class Model:
    def __init__(self, name, ckpt_path):
        self.name = name
        self.ckpt_path = ckpt_path

def main():
    models = [
        Model(name='Pizza3D', ckpt_path=ROOT / 'view_regressor/runs/pizza3d/1ab8hru7/00004999.ckpt'),
    ]
    model_names = [x.name for x in models]
    model_ckpt_paths = [x.ckpt_path for x in models]
    model_name = st.selectbox(
        'Which model to use?',
        model_names,
        index=0
    )
    ckpt_path = model_ckpt_paths[model_names.index(model_name)]
    st.write(f'model = {model_name}: {ckpt_path}')

    device = 'cuda'
    dataset_name = st.selectbox(
        'Dataset',
        options = [
            'LabeledPizza10Subset',
            'Pizza3D',
            'Pizza10',
        ]
    )

    if dataset_name == 'Pizza10':
        dataset = Pizza10Dataset(transform=resnet_transform_train)
    elif dataset_name == 'LabeledPizza10Subset':
        from datasets import utils, true_label_50
        dataset = LabeledPizza10Subset(transform=resnet_transform_train)
        img_dir = Path(f'{ROOT}/data/Pizza10/images/')
        img_names = [f'{x}.jpg' for x in true_label_50.img_index_dict.values()]
    elif dataset_name == 'Pizza3D':
        dataset = Pizza3DDataset(transform=resnet_transform_val)
    else:
        print('not support')
        sys.exit(-1)

    save_dir = Path(f'outputs/model={model_name}/dataset={dataset_name}')
    os.makedirs(save_dir, exist_ok=True)
    outputs_filename = save_dir / 'outputs.pt'
    
    if not os.path.exists(outputs_filename):
        model = load_models(ckpt_path=ckpt_path, device=device)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=False)
        st.write(len(dataset), len(dataloader))
        view_outputs = []
        base_outputs = []
        view_labels = []
        base_labels = []
        batch_idx = 0
        my_bar = st.progress(0)
        
        for img, tgt in dataloader:
            if 'view_label' in tgt:
                view_labels.append(tgt['view_label'])
            if 'base_label' in tgt:
                base_labels.append(tgt['base_label'])
            
            img = img.to(device)
            # img =  normalize(resize(img, size=224))
            output = model(img).cpu()
            view_outputs.append(output[:, :4])
            if 'withBase' in model_name:
                base_outputs.append(output[:, 4:])
            batch_idx += 1
            my_bar.progress(int(batch_idx/len(dataloader)*100))
        
        view_outputs = torch.cat(view_outputs, dim=0)
        view_outputs = denormalize_view_label(view_outputs)
        if 'withBase' in model_name:
            base_outputs = torch.cat(base_outputs, dim=0)
            base_outputs = torch.max(base_outputs, dim=1)[1].view(-1,1)
        
        if 'view_label' in tgt:
            view_labels = torch.cat(view_labels, dim=0)
            view_labels = denormalize_view_label(view_labels)
        if 'base_label' in tgt:
            base_labels = torch.cat(base_labels, dim=0)

        data = {}
        data['view_outputs'] = view_outputs
        data['base_outputs'] = base_outputs
        data['view_labels'] = view_labels
        data['base_labels'] = base_labels

        torch.save(data, outputs_filename)
    else:
        st.write(f'load features from {outputs_filename}')
        data = torch.load(outputs_filename)

    st.write(data['view_outputs'].shape, data['view_labels'].shape)
    
    fig = plt.figure(figsize=(24,5))
    for i in range(4):
        plt.subplot(1,4,i+1)
        _ = plt.hist(data['view_outputs'][:,i].numpy(), bins=30, density=False, fc=(1, 0, 0, 0.5), label='view outputs')
        if torch.is_tensor(data['view_labels']):
            _ = plt.hist(data['view_labels'][:,i].numpy(), bins=30, density=False, fc=(0, 1, 0, 0.5), label='view labels')
        plt.title(f'{Attr.names[i]}')
        plt.legend()
        
        # _ = plt.hist(outputs[:,i], bins=30, density=True)
        # mean, var = stats.distributions.norm.fit(outputs[:,i])
        # x = np.linspace(*Attr.ranges[i], 1000)
        # fitted_data = stats.distributions.norm.pdf(x, mean, var)
        # plt.plot(x, fitted_data, 'r-')
        # plt.title(f'{Attr.names[i]}, {mean:.2f} ({var:.2f})')
    st.pyplot(fig)

    if torch.is_tensor(data['view_labels']):
        # import plotly.figure_factory as ff
        
        from plotly.subplots import make_subplots
        import plotly.graph_objects as go
        fig = make_subplots(rows=1, cols=4)

        import plotly.express as px
        import pandas as pd
        df = pd.DataFrame(
            np.concatenate([data['view_labels'], data['view_outputs']], axis=1), 
            columns=['angle_label', 'scale_label', 'dx_label', 'dy_label', 'angle_pred', 'scale_pred', 'dx_pred', 'dy_pred'])
        df['img_name'] = img_names
        # fig = px.scatter(df, x='angle_label', y='angle_pred', hover_data=['img_name', 'angle_label', 'angle_pred'])
        fig.add_trace(
            go.Scatter(x=df['angle_label'], y=df['angle_pred'], mode="markers", text=df['img_name'], name='angle'),
            row=1, col=1
        )
        fig.update_xaxes(
            title_text="Label",  
            row=1, col=1)
        fig.update_yaxes(
            title_text="Pred", 
            # scaleanchor="x",
            # scaleratio=1,
            row=1, col=1)
 
        fig.add_trace(
            go.Scatter(x=df['scale_label'], y=df['scale_pred'], mode="markers", text=df['img_name'], name='scale'),
            row=1, col=2
        )
        # fig.update_xaxes(
        #     title_text="Label",  
        #     row=1, col=2)
        # fig.update_yaxes(
        #     title_text="Pred", 
        #     scaleanchor="x",
        #     scaleratio=1,
        #     row=1, col=2)

        fig.add_trace(
            go.Scatter(x=df['dx_label'], y=df['dx_pred'], mode="markers", text=df['img_name'], name='dx'),
            row=1, col=3
        )
        # fig.update_xaxes(
        #     title_text="Label",  
        #     row=1, col=3)
        # fig.update_yaxes(
        #     title_text="Pred", 
        #     scaleanchor="x",
        #     scaleratio=1,
        #     row=1, col=3)

        fig.add_trace(
            go.Scatter(x=df['dy_label'], y=df['dy_pred'], mode="markers", text=df['img_name'], name='dy'),
            row=1, col=4
        )
        # fig.update_xaxes(
        #     title_text="Label",  
        #     row=1, col=4)
        # fig.update_yaxes(
        #     title_text="Pred", 
        #     scaleanchor="x",
        #     scaleratio=1,
        #     row=1, col=4)


        # st.plotly_chart(fig, use_container_width=True)

        selected_points = plotly_events(fig)
        if selected_points:
            p = selected_points[0]
            x = p['x']
            y = p['y']
            img_name = img_names[p['pointIndex']]
            st.write(f'Label = {x:.2f}, Pred = {y:.2f}')
            image = Image.open(img_dir/img_name)
            image = image.resize([256, int(256*image.size[1]/image.size[0])])
            st.image(image, caption=img_name)


    if 'withBase' in model_name and torch.is_tensor(data['base_labels']):
        from sklearn.metrics import confusion_matrix
        y_true = data['base_labels']
        y_pred = data['base_outputs']
        st.write(f'Base Prediction Confusion Matrix', confusion_matrix(y_true, y_pred))
        
    
    st.header('Choose view attribute range (from prediction)')
    angle = st.slider("angle", 0.0, 75.0, (0.0, 75.0), 1.0)
    scale = st.slider("scale", 0.5, 2.0, (0.5, 2.0), 0.1)
    dx = st.slider("dx", -112.0, 112.0, (-112.0, 112.0), 1.0)
    dy = st.slider("dy", -112.0, 112.0, (-112.0, 112.0), 1.0)
    ranges = [angle, scale, dx, dy]

    st.write(angle, scale, dx, dy)

    def find_img_idxs():
        img_idxs = set()
        for i in range(4):
            attr = data['view_outputs'][:, i]
            attr_range = ranges[i]
            idxs = set(((attr>=attr_range[0]) & (attr<attr_range[1])).nonzero(as_tuple=False).view(-1).tolist())
            if i == 0:
                img_idxs = idxs
            else:
                img_idxs &= idxs
        if ('withBase' in model_name) and (not show_all):
            if torch.is_tensor(data['base_labels']):
                bases_label = data['base_labels']
                idx = bases_to_show.index(gd_base)
                idxs = set((bases_label==idx).nonzero(as_tuple=False).view(-1).tolist())
                img_idxs &= idxs

            bases_pred = data['base_outputs']
            idx = bases_to_show.index(pred_base)
            idxs = set((bases_pred==idx).nonzero(as_tuple=False).view(-1).tolist())
            img_idxs &= idxs
        return list(img_idxs)

    img_idxs = find_img_idxs()
    st.write('#images found:', len(img_idxs))
    idxs_to_show = img_idxs if len(img_idxs)<64 else np.random.choice(img_idxs, size=64, replace=False)
    if len(idxs_to_show) > 0:
        images = []
        captions = []
        for i in idxs_to_show:
            img, _ = dataset[i]
            images.append(img)
            angle_ = data['view_outputs'][i][0]
            scale_ = data['view_outputs'][i][1]
            dx_ = data['view_outputs'][i][2]
            dy_ = data['view_outputs'][i][3]
            cap = f'a={angle_:.1f}|s={scale_:.1f}|dx={dx_:.1f}|dy={dy_:.1f}'
            if 'withBase' in model_name:
                base_ = data['base_outputs'][i].item()
                cap += f'|base={base_:.1f}'
            captions.append(cap)
                
        images = torch.stack(images, dim=0)
        img_grid = common.make_captioned_image(captions, images, font=10, color=(255,255,255), nrow=8)
        img_grid = img_grid.permute(1, 2, 0)
        st.image(img_grid.numpy())

if __name__ == "__main__":
    main()
