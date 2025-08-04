from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from PIL import Image
import torch
import os
import random

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-seg",  # 可换成 canny、lineart、openpose、scribble 等
    torch_dtype=torch.float16
)

# 加载 SD 主模型（v1.5）
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16
).to("cuda")

#pipe.enable_xformers_memory_efficient_attention()

# ==== 加载图像和 mask ====
img_path = './ijmond_exhaust/manual_negative/'
mask_path = './ijmond_exhaust/vae_outputs/generated/'

img_file = os.path.join(img_path, random.choice(os.listdir(img_path)))
mask_file = os.path.join(mask_path, random.choice(os.listdir(mask_path)))
print("mask_file:", mask_file)

init_image = Image.open(img_file).convert("RGB").resize((512, 512))
mask_image = Image.open(mask_file).convert("RGB").resize((512, 512))
mask_image = mask_image.convert("L")
print("mask_image:", mask_image)

prompt =  "Dense gray smoke only within specified regions, photo, industrial style"
negative_prompt = "no fog, no cloud, no extra structures, no smoke outside mask"


output = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    image=init_image,
    control_image=mask_image,
    num_inference_steps=50,
    guidance_scale=7.5,
    strength=0.5,
    generator=torch.Generator("cuda").manual_seed(42)
)


output_image = output.images[0]
output_image.save("result_controlnet5.jpg")
mask_image.save("used_mask.jpg")

print("✅")