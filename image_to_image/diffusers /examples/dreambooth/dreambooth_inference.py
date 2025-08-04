from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch

unet = UNet2DConditionModel.from_pretrained("./dreambooth-smoke-model/checkpoint-500/unet")

# # if you have trained with `--args.train_text_encoder` make sure to also load the text encoder
# text_encoder = CLIPTextModel.from_pretrained("./dreambooth-smoke-model/checkpoint-500/text_encoder")

pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", unet=unet,dtype=torch.float16,
).to("cuda")

num_images = 5

images = pipeline("A photo of sks smoke ", num_inference_steps=50,
                 guidance_scale=7.5,
                 num_images_per_prompt=num_images
                 ).images

for idx, img in enumerate(images):
    img.save(f"smoke_{idx+1}.png")

print("✅")