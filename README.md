# anonymous-semantic-matching-ddpm

cvpr 2021 paper id 4190

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

You need to modify dataset path, refpath, ckpt, and mode in diffusion.conf file.

```bash
python generate.py --conf diffusion.conf 
```

## Samples

Uncurated samples from semantic level 64.
![git_teaser](https://user-images.githubusercontent.com/74697009/100038137-d3be8900-2e46-11eb-871c-21d0d6e6919c.PNG)

## References

This implementation uses code from following repositories:
- https://github.com/rosinality/denoising-diffusion-pytorch
- https://github.com/assafshocher/PyTorch-Resizer
