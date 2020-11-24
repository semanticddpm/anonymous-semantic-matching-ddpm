# anonymous-semantic-matching-ddpm

cvpr 2021 paper id 4190

## Requirements
If needed, install requirements. I have tested on PyTorch 1.4.
```bash
pip install -r requirements.txt
```

## Training

First prepare lmdb dataset:

```bash
python prepare_data.py --size [SIZES, e.g. 128,256] --out [LMDB NAME] [DATASET PATH]
```

Then run train

```bash
python train.py --conf diffusion.conf 
```

## Generating
First put images and models in folders:
- Create checkpoint folder and put trained model. 
- Put reference images in reference folder.

Then, you need to modify dataset path, refpath, semantic levels, ckpt, and mode in diffusion.conf file.
For example,
```bash
path: data/ffhq_lmdb
refpath: reference/face
n_sample: 10
semantic_level1: 32
ckpt: checkpoint/ffhq_256_1200000.pt
```
Then run generate
```bash
python generate.py --conf diffusion.conf 
```

DDPM trained on FFHQ for 1.2M steps: [GoogleDrive](https://drive.google.com/drive/folders/1aOuHF6yo-IlfidL2duu9IEDHMuQ008vc?usp=sharing)

## Samples

Uncurated samples from semantic level 64.
![git_teaser](https://user-images.githubusercontent.com/74697009/100038137-d3be8900-2e46-11eb-871c-21d0d6e6919c.PNG)

## References

This implementation uses code from following repositories:
- https://github.com/rosinality/denoising-diffusion-pytorch
- https://github.com/assafshocher/PyTorch-Resizer
