This is the official code for **Multi-attribute Pizza Generator (MPG2): Cross-domain Attribute Control with Conditional StyleGAN**.

| [Paper]() | [Demo](http://foodai.cs.rutgers.edu:2022/) |
| --- | --- |

# Setup Environment
**can NOT run on CPU**

```
conda create -n mpg python=3.8
conda activate mpg
git clone git@bitbucket.org:klory/food_project.git
cd food_project
pip install -r requirements.txt
pip install git+https://github.com/pytorch/tnt.git@master
```

# Pretrained models
Pretrained models are stored in [google-link](https://drive.google.com/drive/folders/1_jbSbhhdUPZz-XqxuWnnXzRxLeG-yGln?usp=sharing), files are already in their desired locations, so following the same directory structure will minimize burdens to run the code inside the project (some files are not necessary for the current version of the project as of 2021-03-31).

# Pizza10 dataset

Please follow [MPG repository](https://github.com/klory/MPG_Arxiv).

# Ingredient classifier

Please follow [MPG repository](https://github.com/klory/MPG_Arxiv).

# PizzaView dataset
Download PizzaView Dataset from `google-link/data/Pizza3D`.

cd to `datasets/`

```
$ python pizza3d.py
```

# View regressor

cd to `view_regressor/`

## Train
```
$ CUDA_VISIBLE_DEVICES=0 python train.py --wandb=0
```

## Validate
Download the pretrained model `google-link/view_regressor/runs/pizza3d/1ab8hru7/00004999.ckpt`:
```
$ CUDA_VISIBLE_DEVICES=0 python val.py --ckpt_path=/runs/pizza3d/1ab8hru7/00004999.ckpt
```

# MPG2

cd to `mpg/`,

## Train

```
$ CUDA_VISIBLE_DEVICES=0,1 python train.py --wandb=0
```

## Validate

Download the pretrained model `google-linkmpg/runs/30cupu9m/00260000.ckpt`.

cd to `metrics/`:

```
CUDA_VISIBLE_DEVICES=0 python generate_samples.py --model=mpg
```

----

# Metrics
> cd to `metrics/`,

For more about FID and mAP, follow [MPG repository](https://github.com/klory/MPG_Arxiv).

## FID (Frechet Inception Distance)
To compute FID, we need to first compute the statistics of the real images.

```
CUDA_VISIBLE_DEVICES=0 python calc_inception.py
```

then
```
$ CUDA_VISIBLE_DEVICES=0 python fid.py --model=mpg
```

I got `FID=6.33` using the provided checkpoint.

## mAE (mean Absolute Error) for view attributes

Computing mAE uses the pre-trained view regressor.

```
$ CUDA_VISIBLE_DEVICES=0 python mAE.py --model=mpg
```

# Demo

cd to `metrics/`.

```
CUDA_VISIBLE_DEVICES=0 streamlit run app.py
```