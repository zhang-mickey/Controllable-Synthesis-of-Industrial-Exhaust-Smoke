import os
import numpy as np
from PIL import Image

# 设置路径
img_dir = './ijmond_exhaust/cropped_images/'
mask_dir = './ijmond_exhaust/cropped_masks/'
output_dir = './ijmond_exhaust/masked_output/'
os.makedirs(output_dir, exist_ok=True)

# 获取图像和 mask 文件名（不含扩展名）
img_names = {os.path.splitext(f)[0]: f for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))}
mask_names = {os.path.splitext(f)[0].replace("mask_", ""): f for f in os.listdir(mask_dir) if f.startswith('mask_') and f.endswith(('.jpg', '.png'))}

common_names = set(img_names.keys()) & set(mask_names.keys())

print("common_names:", common_names)

for name in sorted(common_names):
    img_path = os.path.join(img_dir, img_names[name])
    mask_path = os.path.join(mask_dir, mask_names[name])

    image = np.array(Image.open(img_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L"))

    if image.shape[:2] != mask.shape:
        print(f"⚠️ Size mismatch for {name}, skipping...")
        continue

    # mask -> binary -> repeat 3 channel
    binary_mask = (mask > 128).astype(np.uint8)[..., None]
    binary_mask = np.repeat(binary_mask, 3, axis=2)

    # 应用 mask
    masked_image = image * binary_mask

    # 保存结果
    out_path = os.path.join(output_dir, f"{name}_masked.png")
    Image.fromarray(masked_image).save(out_path)

print("✅ 所有图像处理完成。")




