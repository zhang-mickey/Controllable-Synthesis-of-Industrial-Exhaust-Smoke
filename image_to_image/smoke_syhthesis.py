import os
import cv2
import random

import argparse, os, sys, glob
import PIL
import torch
import numpy as np
import matplotlib.pyplot as plt

from omegaconf import OmegaConf
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext

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
    pl_sd = torch.load(ckpt, map_location="cpu")
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


# =======================================================
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

    image = np.array(image)
    mask = np.array(mask)  # .astype(np.float32) / 255.0

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    # mask_out = copy.deepcopy(cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

    # Map color
    R = np.random.uniform(0.7, 0.9)
    G = np.random.uniform(0.2, 0.4)
    B = np.random.uniform(0, 0.1)

    mask[:, :, 0] = mask[:, :, 0] * R
    mask[:, :, 1] = mask[:, :, 1] * G
    mask[:, :, 2] = mask[:, :, 2] * B

    mask = mask / 255.0 + np.random.normal(0, noise_std, size=(w, h, 3))
    mask = img_uint8(mask)

    image = image / 255.0 + mask / 255.0
    image = image * 255
    image[image > 255] = 255
    # image = img_uint8(image)

    mask_out = copy.deepcopy(cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))

    # To tensor
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)

    return (2. * image - 1.), mask_out, [R, G, B]


# =======================================================


def print_text_on_image(image, text, font_path, font_size, text_position, text_color):
    # Open the image
    # image = Image.open(image_path)

    # Create an ImageDraw object
    draw = ImageDraw.Draw(image)

    # Specify the font and size
    font = ImageFont.truetype("arial.ttf", size=font_size)
    # text = 'test'
    # Draw the text on the image
    draw.text(text_position, text, font=font, fill=(255, 255, 255))

    # Save the modified image
    # image.save("image_with_text.jpg")
    return image


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


from datetime import datetime

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

    opt = parser.parse_args()
    # seed_everything(opt.seed)

    # opt.ckpt = "models/ldm/stable-diffusion-v1/sd-v1-1.ckpt"
    opt.ckpt = "models/ldm/stable-diffusion-v1/v1-5-pruned.ckpt"

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # =======================================================
    # All exp files will be saved into this {exp_name} folder
    exp_name = 'perlin_test'
    num_sample = 10
    image_info = []  # all image infos will be saved to this numpy array

    '''
    Here is the main setting for diffuser:
        strength to 0.99 means fully random, 0.3-0.5 is a good range for quality
        opt.ddim_steps is set to 10 by default for speed and quality balance
        noise_std is the post-random noise added to the image, see paper for more details.

        You can put them into the loop for experiments, see line 280-285
    '''
    opt.scale = 5
    opt.strength = 0.5
    opt.ddim_steps = int(10 / (opt.strength))
    noise_std = 0

    img_path = './ijmond_exhaust/cropped_images/'

    # mask_path = './exp/mask/'
    mask_path = './ijmond_exhaust/cropped_mask/'
    # =======================================================

    opt.prompt = "Industrial toxic exhaust in snow, flame and smoke view, photo realistic, high resolution, 4k, HD"
    opt.n_samples = 1
    opt.n_iter = 1
    opt.skip_grid = True
    opt.watermark = False
    opt.outdir = f'./outputs/{exp_name}/'

    outpath = opt.outdir
    os.makedirs(outpath, exist_ok=True)
    mask_save_path = outpath + '/masks/'
    os.makedirs(mask_save_path, exist_ok=True)

    img_list = os.listdir(img_path)
    mask_list = os.listdir(mask_path)

    precision_scope = autocast if opt.precision == "autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                tic = time.time()
                all_samples = list()
                i = 0
                while i < num_sample:
                    try:
                        i += 1

                        img_file = img_path + random.choice(img_list)
                        image_name = img_file.split('/')[-1].split('.')[0]
                        mask_file = mask_path + random.choice(mask_list)
                        mask_name = mask_file.split('/')[-1].split('.')[0]
                        mask = Image.open(mask_file).convert("L")
                        image = Image.open(img_file).convert("RGB")

                        opt.init_img = img_file

                        # =======================================================
                        # # For experiments
                        # opt.scale = random.randint(5, 15)
                        # opt.strength = random.uniform(0.3, 0.7)
                        # noise_std = 0 #random.uniform(0, 0.1)
                        # opt.ddim_steps = int(10/(opt.strength))
                        # =======================================================

                        if opt.plms:
                            raise NotImplementedError("PLMS sampler not (yet) supported")
                            sampler = PLMSSampler(model)
                        else:
                            sampler = DDIMSampler(model)

                        batch_size = opt.n_samples
                        n_rows = opt.n_rows if opt.n_rows > 0 else batch_size
                        if not opt.from_file:
                            prompt = opt.prompt
                            assert prompt is not None
                            data = [batch_size * [prompt]]

                        else:
                            print(f"reading prompts from {opt.from_file}")
                            with open(opt.from_file, "r") as f:
                                data = f.read().splitlines()
                                data = list(chunk(data, batch_size))

                        sample_path = os.path.join(outpath, "samples")
                        os.makedirs(sample_path, exist_ok=True)
                        base_count = len(os.listdir(sample_path))
                        grid_count = len(os.listdir(outpath)) - 1

                        assert os.path.isfile(opt.init_img)
                        init_image, mask_out, [R, G, B] = load_img(image, opt.init_img, mask, noise_std)
                        init_image = init_image.to(device)
                        init_image = repeat(init_image, '1 ... -> b ...', b=batch_size)
                        init_latent = model.get_first_stage_encoding(
                            model.encode_first_stage(init_image))  # move to latent space

                        # init_latent_image = latent2img(init_latent)
                        # init_latent_image = cv2.resize(cv2.cvtColor(init_latent_image, cv2.COLOR_BGR2RGB), (512,512))
                        # plt.imshow(init_latent_image)

                        assert 0. <= opt.strength <= 1., 'can only work with strength in [0.0, 1.0]'
                        t_enc = int(opt.strength * opt.ddim_steps)
                        print(f"target t_enc is {t_enc} steps")

                        try:
                            sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)
                        except:
                            opt.ddim_steps = 100
                            sampler.make_schedule(ddim_num_steps=opt.ddim_steps, ddim_eta=opt.ddim_eta, verbose=False)

                        for n in trange(opt.n_iter, desc="Sampling"):

                            for prompts in tqdm(data, desc="data"):
                                uc = None
                                if opt.scale != 1.0:
                                    uc = model.get_learned_conditioning(batch_size * [""])
                                if isinstance(prompts, tuple):
                                    prompts = list(prompts)
                                c = model.get_learned_conditioning(prompts)

                                # encode (scaled latent)
                                z_enc = sampler.stochastic_encode(init_latent,
                                                                  torch.tensor([t_enc] * batch_size).to(device))
                                # decode it
                                samples, cc, mid = sampler.decode(z_enc, c, t_enc,
                                                                  unconditional_guidance_scale=opt.scale,
                                                                  unconditional_conditioning=uc, )

                                x_samples = model.decode_first_stage(samples)
                                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                                if not opt.skip_save:
                                    x_sample_i = 1
                                    for x_sample in x_samples:
                                        x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')

                                        image = Image.fromarray(x_sample.astype(np.uint8))
                                        # image = normalize_image(image)
                                        strength = round(opt.strength, 2)
                                        noise = round(noise_std, 2)
                                        font_size = 20

                                        if opt.watermark == True:
                                            image = print_text_on_image(image,
                                                                        f"scale={opt.scale} strength={strength} steps={opt.ddim_steps} noise={noise}",
                                                                        "arial.ttf", font_size, (10, 10),
                                                                        (255, 255, 255))
                                            image = print_text_on_image(image, f"{opt.prompt}", "arial.ttf", font_size,
                                                                        (10, 30), (255, 255, 255))
                                            image = print_text_on_image(image, f"{img_file}", "arial.ttf", font_size,
                                                                        (10, 50), (255, 255, 255))

                                        # image_save_name = f"{image_name}_{i}_{x_sample_i}{n}.jpg"
                                        image_save_name = f"{image_name}_{mask_name}_{i}_{x_sample_i}{n}.jpg"
                                        image.save(os.path.join(sample_path, image_save_name))
                                        # cv2.imwrite(os.path.join(sample_path, f"{image_name}_{i}_init_latent_image.png"), init_latent_image)

                                        base_count += 1
                                        x_sample_i += 1
                                all_samples.append(x_samples)

                        # mask_save_name = f"{image_name}_{i}_{mask_name}.png"
                        mask_save_name = f"{image_name}_{mask_name}_{i}.jpg"
                        cv2.imwrite(os.path.join(mask_save_path, mask_save_name), mask_out)

                        if not opt.skip_grid:
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
                                           opt.prompt,
                                           noise_std,
                                           opt.scale,
                                           opt.strength,
                                           opt.ddim_steps,
                                           R, G, B
                                           ])

                    except:
                        print('skip...')

            print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
                  f" \nEnjoy.")

            image_info = np.array(image_info, dtype=object)
            np.save(f'{outpath}/dataset.npy', image_info)
            print("image_info has been saved as dataset.npy")












