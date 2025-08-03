import os
import cv2
import random

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from datetime import datetime

import time
import copy
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms

from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def normalize_image(image):
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    normalized_image = (image / 1.0 - mean) / std
    return normalized_image


def add_gaussian_noise(tensor, mean, std):
    noise = torch.randn_like(torch.Tensor(np.array(tensor))) * std + mean
    return tensor + noise.numpy()


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    print(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


# def get_smoke_color_scheme(mode="random"):
#     if mode == "gray":
#         val = np.random.uniform(0.7, 0.9)
#         return val, val, val
#     elif mode == "black":
#         val = np.random.uniform(0.0, 0.2)
#         return val, val, val
def get_smoke_color_from_image(image_np, mask_np, strategy="mean"):
    """
    Args:
        image_np: np.ndarray, shape (H, W, 3), RGB
        mask_np: np.ndarray, shape (H, W), grayscale mask [0,255]
    Returns:
        (r, g, b): sampled color tuple
    """
    binary_mask = (mask_np > 128)
    if not binary_mask.any():
        return (200, 200, 200)  # fallback to light gray if mask is empty

    pixels = image_np[binary_mask]  # shape: (N, 3)

    if strategy == "mean":
        rgb = pixels.mean(axis=0)
    elif strategy == "median":
        rgb = np.median(pixels, axis=0)
    elif strategy == "random":
        rgb = pixels[np.random.randint(0, len(pixels))]
    else:
        raise ValueError("Unknown strategy")

    return tuple(rgb.astype(np.uint8))


def apply_sampled_color(mask_np, color_rgb):
    """
    Args:
        mask_np: (H, W) grayscale mask
        color_rgb: (R, G, B) tuple
    Returns:
        colored mask (H, W, 3)
    """
    h, w = mask_np.shape
    colored_mask = np.ones((h, w, 3), dtype=np.uint8) * np.array(color_rgb, dtype=np.uint8)
    return colored_mask


def apply_colormap_to_mask(mask, colormap_name='bone'):
    colormap = cm.get_cmap(colormap_name)
    mask_norm = mask / 255.0  # normalize to [0, 1]
    colored_mask = (colormap(mask_norm)[..., :3] * 255).astype(np.uint8)
    return colored_mask


def blend_image_and_mask(image, colored_mask, mask_alpha):
    return (image * (1 - mask_alpha[..., None]) + colored_mask * mask_alpha[..., None]).astype(np.uint8)


def load_img(image, path, mask, noise_std):
    # z = len(image.mode)
    w, h = mask.size
    w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 32
    ratio = w / h
    if w > h:
        w = 512
        h = int(w / ratio)
        if h % 64 != 0:
            h = int((h // 64 + 1) * 64)
    else:
        h = 512
        w = int(h * ratio)
        if w % 64 != 0:
            w = int((w // 64 + 1) * 64)
    print(f"loaded input image from {path}, resize to ({w}, {h}) ")
    image = image.resize((w, h), resample=PIL.Image.LANCZOS)
    mask = mask.resize((w, h), resample=PIL.Image.LANCZOS)

    image_np = np.array(image).astype(np.float32)
    mask_np = np.array(mask).astype(np.float32)

    mask = cv2.cvtColor(mask_np, cv2.COLOR_GRAY2RGB)

    sampled_color = get_smoke_color_from_image(image_np, mask_np, strategy="mean")
    colored_mask = apply_sampled_color(mask_np, sampled_color)

    alpha = mask_np / 255.0

    noisy_mask = colored_mask + np.random.normal(0, noise_std, colored_mask.shape)
    noisy_mask = np.clip(noisy_mask, 0, 255).astype(np.uint8)
    blended = blend_image_and_mask(image_np, noisy_mask, alpha)

    mask_out = copy.deepcopy(cv2.cvtColor(noisy_mask, cv2.COLOR_RGB2BGR))

    image_tensor = torch.from_numpy((blended / 255.0)[None].transpose(0, 3, 1, 2).astype(np.float32))
    return (2. * image_tensor - 1.), mask_out, [None, None, None]


def img_uint8(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image


def tensor2img(x):
    # x = init_image
    image = x.detach().cpu().squeeze(0).permute(1, 2, 0).numpy()
    return img_uint8(image)


def latent2img(z):
    v1_4_rgb_latent_factors = [
        #   R       G       B
        [0.298, 0.207, 0.208],  # L1
        [0.187, 0.286, 0.173],  # L2
        [-0.158, 0.189, 0.264],  # L3
        [-0.184, -0.271, -0.473],  # L4
    ]
    latent_factor = torch.tensor(v1_4_rgb_latent_factors).to(device)
    latent_image = z[0].permute(1, 2, 0) @ latent_factor
    latent_image = latent_image.cpu().numpy()
    return img_uint8(latent_image)


if __name__ == "__main__":
    data_label = {"image:"}
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, nargs="?", default="a painting of a virus monster playing guitar",
                        help="the prompt to render")
    parser.add_argument("--init-img", type=str, nargs="?", help="path to the input image", default="test2.jpg")
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to",
                        default="outputs/img2img-samples")
    parser.add_argument("--skip_grid", action='store_true',
                        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples", )
    parser.add_argument("--skip_save", action='store_true',
                        help="do not save indiviual samples. For speed measurements.", )

    parser.add_argument("--ddim_steps", type=int, default=50, help="number of ddim sampling steps", )

    parser.add_argument("--plms", action='store_true', help="use plms sampling", )
    parser.add_argument("--fixed_code", action='store_true',
                        help="if enabled, uses the same starting code across all samples ", )
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often", )
    parser.add_argument("--C", type=int, default=4, help="latent channels", )
    parser.add_argument("--f", type=int, default=8, help="downsampling factor, most often 8 or 16", )

    parser.add_argument("--n_samples", type=int, default=2,
                        help="how many samples to produce for each given prompt. A.k.a batch size", )

    parser.add_argument("--n_rows", type=int, default=0, help="rows in the grid (default: n_samples)", )
    parser.add_argument("--scale", type=float, default=5.0,
                        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))", )
    parser.add_argument("--strength", type=float, default=0.75,
                        help="strength for noising/unnoising. 1.0 corresponds to full destruction of information in init image", )
    parser.add_argument("--from-file", type=str, help="if specified, load prompts from this file", )
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml",
                        help="path to config which constructs model", )
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/sd-v1-1.ckpt",
                        help="path to checkpoint of model", )
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)", )
    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"],
                        default="autocast")

    args = parser.parse_args()

    args.ckpt = "models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt"

    config = OmegaConf.load(f"{args.config}")
    model = load_model_from_config(config, f"{args.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    exp_name = 'smoke_synthesis'
    num_sample = 10
    image_info = []  # all image infos will be saved to this numpy array

    '''
    strength to 0.99 means fully random, 0.3-0.5 is a good range for quality
    ddim_steps is set to 10 by default for speed and quality balance
    noise_std is the post-random noise added to the image
    '''
    args.scale = 5

    args.strength = 0.4
    args.ddim_steps = int(10 / (args.strength))
    noise_std = 10

    img_path = './ijmond_exhaust/manual_negative/'
    mask_path = './ijmond_exhaust/cropped_images/'
    random_mask_path = './ijmond_exhaust/vae_outputs/generated/'
    # =======================================================

    args.prompt = "Industrial factory with chimneys and clear blue sky, 4k, HD"
    # A large industrial facility with smokestacks is visible under a blue sky with scattered clouds.
    args.n_samples = 1
    args.n_iter = 1
    args.skip_grid = True
    args.outdir = f'./outputs/{exp_name}/'

    outpath = args.outdir
    os.makedirs(outpath, exist_ok=True)
    mask_save_path = outpath + '/masks/'
    os.makedirs(mask_save_path, exist_ok=True)

    img_list = os.listdir(img_path)
    mask_list = os.listdir(random_mask_path)

    precision_scope = autocast if args.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                i = 0
                while i < num_sample:
                    if True:
                        i += 1

                        img_file = img_path + random.choice(img_list)
                        image_name = img_file.split('/')[-1].split('.')[0]
                        mask_file = random_mask_path + random.choice(mask_list)
                        mask_name = mask_file.split('/')[-1].split('.')[0]
                        mask = Image.open(mask_file).convert("L")
                        image = Image.open(img_file).convert("RGB")

                        args.init_img = img_file
                        sampler = DDIMSampler(model)

                        batch_size = args.n_samples
                        n_rows = args.n_rows if args.n_rows > 0 else batch_size
                        if not args.from_file:
                            prompt = args.prompt
                            assert prompt is not None
                            data = [batch_size * [prompt]]

                        else:
                            print(f"reading prompts from {args.from_file}")
                            with open(args.from_file, "r") as f:
                                data = f.read().splitlines()
                                data = list(chunk(data, batch_size))

                        sample_path = os.path.join(outpath, "samples")
                        os.makedirs(sample_path, exist_ok=True)
                        base_count = len(os.listdir(sample_path))
                        grid_count = len(os.listdir(outpath)) - 1

                        assert os.path.isfile(args.init_img)
                        init_image, mask_out, [R, G, B] = load_img(image, args.init_img, mask, noise_std)
                        init_image = init_image.to(device)
                        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                        init_latent = model.get_first_stage_encoding(
                            model.encode_first_stage(init_image))  # move to latent space

                        assert 0. <= args.strength <= 1., 'can only work with strength in [0.0, 1.0]'
                        t_enc = int(args.strength * args.ddim_steps)
                        print(f"target t_enc is {t_enc} steps")

                        sampler.make_schedule(ddim_num_steps=args.ddim_steps, ddim_eta=args.ddim_eta, verbose=False)

                        for n in trange(args.n_iter, desc="Sampling"):

                            for prompts in tqdm(data, desc="data"):
                                uc = None
                                if args.scale != 1.0:
                                    uc = model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = model.get_learned_conditioning(prompts)

                                # encode (scaled latent)
                                z_enc = sampler.stochastic_encode(init_latent,
                                                                  torch.tensor([t_enc] * batch_size).to(device))
                                # decode
                                samples, cc, mid = sampler.decode(z_enc, c, t_enc,
                                                                  unconditional_guidance_scale=args.scale,
                                                                  unconditional_conditioning=uc, )

                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                if not args.skip_save:
                                    x_sample_i = 1
                                    for x_sample in x_samples:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')

                                        image = Image.fromarray(x_sample.astype(np.uint8))
                                        # image = normalize_image(image)
                                        strength = round(args.strength, 2)
                                        noise = round(noise_std, 2)
                                        # image_save_name = f"{image_name}_{i}_{x_sample_i}{n}.jpg"
                                        image_save_name = f"{image_name}_{mask_name}_{i}_{x_sample_i}{n}.jpg"
                                        image.save(os.path.join(sample_path, image_save_name))
                                        # cv2.imwrite(os.path.join(sample_path, f"{image_name}_{i}_init_latent_image.png"), init_latent_image)

                                        base_count += 1
                                        x_sample_i += 1
                                all_samples.append(x_samples)

                        mask_save_name = f"{image_name}_{mask_name}_{i}.jpg"
                        cv2.imwrite(os.path.join(mask_save_path, mask_save_name), mask_out)

                        if not args.skip_grid:
                            # additionally, save as grid
                            grid = torch.stack(all_samples, 0)
                            grid = rearrange(grid, 'n b c h w -> (n b) c h w')
                            grid = make_grid(grid, nrow=n_rows)

                            # to image
                            grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
                            Image.fromarray(grid.astype(np.uint8)).save(
                                os.path.join(outpath, f'grid-{grid_count:04}.png'))
                            grid_count += 1

                        toc = time.time()

                        current_datetime = datetime.now()
                        date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

                        image_info.append([i, img_file, mask_file,
                                           image_save_name,
                                           mask_save_name,
                                           date_time_string,
                                           args.prompt,
                                           noise_std,
                                           args.scale,
                                           args.strength,
                                           args.ddim_steps,
                                           R, G, B
                                           ])

            print(f"samples: \n{outpath} \n")
            image_info = np.array(image_info, dtype=object)
            np.save(f'{outpath}/dataset.npy', image_info)
            print("image_info has been saved as dataset.npy")
