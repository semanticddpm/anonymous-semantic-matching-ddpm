import os

import torch
from torch import nn, optim
from torch.utils import data
from torch.nn import functional as F
import torchvision
from torchvision import transforms
from torchvision import transforms, utils
from tensorfn import load_arg_config
from tensorfn import distributed as dist
import numpy as np

from model import UNet
from diffusion import GaussianDiffusion, make_beta_schedule
from config import DiffusionConfig

from PIL import Image
from tqdm import tqdm
import random
from resizer import Resizer
import numpy as np


def sample_data(loader):
    loader_iter = iter(loader)
    epoch = 0

    while True:
        try:
            yield epoch, next(loader_iter)

        except StopIteration:
            epoch += 1
            loader_iter = iter(loader)

            yield epoch, next(loader_iter)


def generate(conf, ema, diffusion, device):

    path = conf.dataset.refpath
    paths = list(os.listdir(path))
    paths = sorted(paths)

    pbar = range(len(paths))
    pbar = tqdm(pbar, initial=0, dynamic_ncols=True, smoothing=0.01)

    index = list(range(len(paths)))
    random.shuffle(index)

    resolution = conf.dataset.resolution
    n_sample = conf.evaluate.n_sample
    semantic_level1 = conf.evaluate.semantic_level1
    semantic_level2 = conf.evaluate.semantic_level2

    shape = (n_sample, 3, resolution, resolution)
    shape_a = (n_sample * 5, 3, resolution, resolution)

    lr_shape = (n_sample, 3, int(resolution / semantic_level1), int(resolution / semantic_level1))
    downc = Resizer(shape, 1 / semantic_level1, kernel="cubic").to(device)
    upc = Resizer(lr_shape, semantic_level1, kernel="cubic").to(device)
    resizers1 = (upc, downc)

    if conf.evaluate.mode in ['scribble']:
        lr_shape = (n_sample, 3, int(resolution / semantic_level2), int(resolution / semantic_level2))
        down = Resizer(shape, 1 / semantic_level2).to(device)
        up = Resizer(lr_shape, semantic_level2).to(device)
        resizers2 = (up, down)

    with torch.no_grad():
        ema.eval()

        for i in pbar:

            img = np.asarray(Image.open(path + '/' + paths[i]))

            if img.shape[2] == 4:
                img = img[:, :, :3]
            im = img.transpose((2, 0, 1))
            img = torch.tensor(np.asarray(im, dtype=np.float32), device='cpu', requires_grad=True).cuda() / 127.5 - 1.
            img = F.interpolate(img.unsqueeze(0), size=(resolution, resolution),mode="bicubic", align_corners=True)

            img = img.to(device).repeat(n_sample, 1, 1, 1)

            if conf.evaluate.mode not in ['scribble']:
                sample = diffusion.p_sm_loop(img, resizers1, ema, shape, device)

            else:
                img2 = np.asarray(Image.open(path + '/' + paths[2 * i]))  # modify this path
                if img.shape[2] == 4:
                    img2 = img2[:, :, :3]
                im2 = img2.transpose((2, 0, 1))
                img2 = torch.tensor(np.asarray(im2, dtype=np.float32), device='cpu',requires_grad=True).cuda() / 127.5 - 1.
                img2 = F.interpolate(img2.unsqueeze(0), size=(resolution, resolution),mode="bicubic", align_corners=True)

                img2 = img2.to(device).repeat(n_sample, 1, 1, 1)
                t = 250
                sample = diffusion.p_harmonize_loop((img, img2), resizers1, resizers2, ema, shape, device, t)

            for j in range(n_sample):
                utils.save_image(
                    sample[j].unsqueeze(0),
                    f'output/{conf.evaluate.mode}/ID{i}_{j}.png',
                    nrow=1,
                    normalize=True,
                    range=(-1, 1)
                )


def main(conf):
    device = "cuda"
    beta_schedule = "linear"
    beta_start = 1e-4
    beta_end = 2e-2
    n_timestep = 1000

    ckpt_diff = conf.evaluate.ckpt

    conf.distributed = False

    ema = conf.model.make()
    ema = ema.to(device)

    print(f'load model from: {conf.evaluate.ckpt}')
    ckpt = torch.load(conf.evaluate.ckpt, map_location=lambda storage, loc: storage)
    ema.load_state_dict(ckpt["ema"])

    betas = conf.diffusion.beta_schedule.make()

    # betas: 0.0001 --> 0.02
    diffusion = GaussianDiffusion(betas).to(device)

    generate(conf, ema, diffusion, device)


if __name__ == "__main__":
    conf = load_arg_config(DiffusionConfig)

    dist.launch(
        main, conf.n_gpu, conf.n_machine, conf.machine_rank, conf.dist_url, args=(conf,)
    )
