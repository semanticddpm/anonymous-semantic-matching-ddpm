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
[git_teaser.pdf](https://github.com/semanticddpm/anonymous-semantic-matching-ddpm/files/5586948/git_teaser.pdf)

## References

This implementation uses code from following repositories:
- https://github.com/rosinality/denoising-diffusion-pytorch
- https://github.com/assafshocher/PyTorch-Resizer
