import math
import numbers

import torch
from torch import nn
from torch.nn import functional as F
from tensorfn.config import config_model

class GaussianSmoothing(nn.Module):

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):

        return self.conv(input, weight=self.weight, groups=self.groups)


@config_model()
def make_beta_schedule(
    schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3
):
    if schedule == "linear":
        betas = (
            torch.linspace(
                linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64
            )
            ** 2
        )

    elif schedule == "cosine":
        timesteps = (
            torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * math.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = betas.clamp(max=0.999)

    return betas


def extract(input, t, shape):
    out = torch.gather(input, 0, t)
    reshape = [shape[0]] + [1] * (len(shape) - 1)
    out = out.reshape(*reshape)

    return out


def noise_like(shape, noise_fn, device, repeat=False):
    if repeat:
        resid = [1] * (len(shape) - 1)
        shape_one = (1, *shape[1:])

        return noise_fn(*shape_one, device=device).repeat(shape[0], *resid)

    else:
        return noise_fn(*shape, device=device)


class GaussianDiffusion(nn.Module):
    def __init__(self, betas):
        super().__init__()

        betas = betas.type(torch.float64)
        timesteps = betas.shape[0]
        self.num_timesteps = int(timesteps)

        alphas = 1 - betas
        alphas_cumprod = torch.cumprod(alphas, 0)
        alphas_cumprod_prev = torch.cat(
            (torch.tensor([1], dtype=torch.float64), alphas_cumprod[:-1]), 0
        )
        posterior_variance = betas * (1 - alphas_cumprod_prev) / (1 - alphas_cumprod)

        self.register("betas", betas)
        self.register("alphas_cumprod", alphas_cumprod)
        self.register("alphas_cumprod_prev", alphas_cumprod_prev)

        self.register("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register("sqrt_one_minus_alphas_cumprod", torch.sqrt(1 - alphas_cumprod))
        self.register("log_one_minus_alphas_cumprod", torch.log(1 - alphas_cumprod))
        self.register("sqrt_recip_alphas_cumprod", torch.rsqrt(alphas_cumprod))
        self.register("sqrt_recipm1_alphas_cumprod", torch.sqrt(1 / alphas_cumprod - 1))
        self.register("posterior_variance", posterior_variance)
        self.register(
            "posterior_log_variance_clipped",
            torch.log(posterior_variance.clamp(min=1e-20)),
        )
        self.register(
            "posterior_mean_coef1",
            (betas * torch.sqrt(alphas_cumprod_prev) / (1 - alphas_cumprod)),
        )
        self.register(
            "posterior_mean_coef2",
            ((1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - alphas_cumprod)),
        )

    def register(self, name, tensor):
        self.register_buffer(name, tensor.type(torch.float32))

    def q_sample(self, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        return (
            extract(self.sqrt_alphas_cumprod, t, x_0.shape) * x_0
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape) * noise
        )

    def p_loss(self, model, x_0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        x_noise = self.q_sample(x_0, t, noise)
        x_recon = model(x_noise, t)

        return F.mse_loss(x_recon, noise)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_0, x_t, t):
        mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_0
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        var = extract(self.posterior_variance, t, x_t.shape)
        log_var_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)

        return mean, var, log_var_clipped

    def p_mean_variance(self, model, x, t, clip_denoised):
        x_recon = self.predict_start_from_noise(x, t, noise=model(x, t))

        if clip_denoised:
            x_recon = x_recon.clamp(min=-1, max=1)

        mean, var, log_var = self.q_posterior(x_recon, x, t)

        return mean, var, log_var

    def p_sample(self, model, x, t, noise_fn, clip_denoised=True, repeat_noise=False):
        mean, _, log_var = self.p_mean_variance(model, x, t, clip_denoised)
        noise = noise_like(x.shape, noise_fn, x.device, repeat_noise)
        shape = [x.shape[0]] + [1] * (x.ndim - 1)
        nonzero_mask = (1 - (t == 0).type(torch.float32)).view(*shape)

        return mean + nonzero_mask * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, model, shape, device, noise_fn=torch.randn):
        img = noise_fn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
            )

        return img

    @torch.no_grad()
    def p_sm_loop(self, img, resizers, model, shape, device, noise_fn=torch.randn):
        up, down = resizers
        x0 = img
        img = noise_like(shape, noise_fn, device, repeat=False)

        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
            )

            # latent semantic matching
            time = torch.full((shape[0],), i, dtype=torch.int64).to(device)
            img = up(down(self.q_sample(x0, time, noise_fn(shape, device=device)))) + img - up(down(img))

        return img

    # ablation on LPF kernels
    @torch.no_grad()
    def p_ablation_loop(self, img, resizers, model, shape, device, noise_fn=torch.randn):
        resizersc, resizersl2, resizersl3, resizersb, resizersl = resizers
        upc, downc = resizersc
        upl2, downl2 = resizersl2
        upl3, downl3 = resizersl3
        upb, downb = resizersb
        upl, downl = resizersl

        x0 = img.repeat(5, 1, 1, 1)
        img = noise_like(shape, noise_fn, device, repeat=True)

        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
                repeat_noise=True,
            )

            # latent semantic matching
            time = torch.full((shape[0],), i, dtype=torch.int64).to(device)
            gt = self.q_sample(x0, time, noise_like(shape, noise_fn, device, repeat=True))
            gtc, gtl2, gtl3, gtb, gtl = torch.chunk(gt, 5, 0)
            imgc, imgl2, imgl3, imgb, imgl = torch.chunk(img, 5, 0)

            imgc = upc(downc(gtc)) + imgc - upc(downc(imgc))
            imgl2 = upl2(downl2(gtl2)) + imgl2 - upl2(downl2(imgl2))
            imgl3 = upl3(downl3(gtl3)) + imgl3 - upl3(downl3(imgl3))
            #imgb = upb(downb(gtb)) + imgb - upb(downb(imgb))
            imgl = upl(downl(gtl)) + imgl - upl(downl(imgl))

            img = torch.cat([imgc, imgl2, imgl3, imgb, imgl], 0)

        return img

    @torch.no_grad()
    def p_harmonize_loop(self, img, resizers, resizers2, model, shape, device, t=250, noise_fn=torch.randn):
        up, down = resizers
        up2, down2 = resizers2
        x0, scribble = img
        img = noise_fn(shape, device=device)

        for i in reversed(range(self.num_timesteps)):
            img = self.p_sample(
                model,
                img,
                torch.full((shape[0],), i, dtype=torch.int64).to(device),
                noise_fn=noise_fn,
            )

            time = torch.full((shape[0],), i, dtype=torch.int64).to(device)
            if i > t:
                img = up(down(self.q_sample(scribble, time, noise_fn(shape, device=device)))) + img - up(down(img))
            else:
                img = up2(down2(self.q_sample(x0, time, noise_fn(shape, device=device)))) + img - up2(down2(img))

        return img