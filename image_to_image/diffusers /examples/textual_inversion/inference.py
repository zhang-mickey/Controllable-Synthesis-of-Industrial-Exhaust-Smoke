from diffusers import DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import torch
import os
from PIL import Image
from torch_fidelity import calculate_metrics
from safetensors.torch import load_file
from transformers import CLIPTextModel, CLIPTokenizer



pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",dtype=torch.float16,
).to("cuda")


embedding_path = "./textual-inversion-smoke/learned_embeds-steps-1000.safetensors"
loaded_embeds = load_file(embedding_path)  # { "<sks>": tensor([...]) }
tokenizer: CLIPTokenizer = pipeline.tokenizer
text_encoder: CLIPTextModel = pipeline.text_encoder

for token, embed in loaded_embeds.items():
    print(f"Injecting token: {token}")
    num_added = tokenizer.add_tokens(token)
    if num_added == 0:
        print(f"Token {token} already in tokenizer.")

    text_encoder.resize_token_embeddings(len(tokenizer))

    token_id = tokenizer.convert_tokens_to_ids(token)
    with torch.no_grad():
        text_encoder.get_input_embeddings().weight[token_id] = embed

num_images = 100

save_dir = "./generated_images"
os.makedirs(save_dir, exist_ok=True)


images = pipeline("A photo of <sks-smoke> ", num_inference_steps=50,
                 guidance_scale=7.5,
                 num_images_per_prompt=num_images
                 ).images

for idx, img in enumerate(images):
    img.save(os.path.join(save_dir, f"smoke_{idx+400}.png"))

print("✅")


def resize_all_images(folder, size=(512, 512)):
    for filename in os.listdir(folder):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            path = os.path.join(folder, filename)
            try:
                img = Image.open(path).convert("RGB")
                img = img.resize(size, Image.BILINEAR)
                img.save(path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")

resize_all_images("./manual_positive", size=(256, 256))
resize_all_images("./generated_images", size=(256, 256))


metrics = calculate_metrics(
    input1="./manual_positive",         # real
    input2="./generated_images",        # generated
    cuda=True,
    isc=False,
    fid=True,
    verbose=False
)

print("FID:", metrics["frechet_inception_distance"])


