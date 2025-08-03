import os
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import save_image



def weighted_bce_loss(recon_x, x, pos_weight=5.0):
    bce = F.binary_cross_entropy(recon_x, x, reduction='none')
    weight = torch.where(x == 1, pos_weight, 1.0)
    return (bce * weight).sum()


# ---------------------- Mask Preprocessing ----------------------
def preprocess_masks(input_dir, output_dir, size=(512, 512)):
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(input_dir):
        if fname.endswith(('.png', '.jpg')):
            img = Image.open(os.path.join(input_dir, fname)).convert('L')
            img = img.resize(size, Image.BILINEAR)
            bin_mask = (np.array(img) > 128).astype(np.uint8) * 255
            Image.fromarray(bin_mask).save(os.path.join(output_dir, fname))


# ---------------------- Custom Mask Dataset ----------------------
class MaskDataset(Dataset):
    def __init__(self, mask_dir, size=(512, 512)):
        self.mask_dir = mask_dir
        self.mask_files = [f for f in os.listdir(mask_dir) if f.endswith(('.png', '.jpg'))]
        self.transform = transforms.Compose([
            transforms.Resize(size),
            transforms.ToTensor(),  # Converts to [0,1] float32, shape (1,H,W)
        ])

    def __len__(self):
        return len(self.mask_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.mask_dir, self.mask_files[idx])
        mask = Image.open(img_path).convert("L")
        mask = self.transform(mask)
        return mask


class VAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),  # 512->256
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 256->128
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),# 128->64
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),# 64->32
            nn.ReLU(),
            nn.Flatten()
        )
        self.fc_mu = nn.Linear(256 * 32 * 32, latent_dim)
        self.fc_logvar = nn.Linear(256 * 32 * 32, latent_dim)
        self.fc_decode = nn.Linear(latent_dim, 256 * 32 * 32)

        self.decoder = nn.Sequential(
            nn.Unflatten(1, (256, 32, 32)),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 32->64
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),   # 64->128
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),    # 128->256
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),     # 256->512
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_decode(z)
        return self.decoder(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decode(z)
        return x_hat, mu, logvar

# ---------------------- Train & Generate ----------------------
def train_vae():
    # Parameters
    raw_mask_dir = './ijmond_exhaust/masked_output/'
    proc_mask_dir = './ijmond_exhaust/preprocessed_masks/'
    save_dir = './ijmond_exhaust/vae_outputs/'
    os.makedirs(save_dir, exist_ok=True)
    batch_size = 32
    latent_dim = 32
    num_epochs = 30
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Step 1: Preprocess
    preprocess_masks(raw_mask_dir, proc_mask_dir)

    # Step 2: Load dataset
    dataset = MaskDataset(proc_mask_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Step 3: Init model
    vae = VAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)

    def vae_loss(recon_x, x, mu, logvar):
        # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        BCE = weighted_bce_loss(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return BCE + KLD

    # Step 4: Train
    for epoch in range(1, num_epochs + 1):
        for batch in loader:
            batch = batch.to(device)
            x_hat, mu, logvar = vae(batch)
            loss = vae_loss(x_hat, batch, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch}/{num_epochs}] Loss: {loss.item():.2f}")

        if epoch % 5 == 0:
            save_image(x_hat[:16], os.path.join(save_dir, f'recon_epoch_{epoch}.png'), nrow=4)

    # Step 5: Generate new masks
    vae.eval()
    gen_dir = os.path.join(save_dir, 'generated')
    os.makedirs(gen_dir, exist_ok=True)

    with torch.no_grad():
        for i in range(20):
            z = torch.randn(1, latent_dim).to(device)
            gen_mask = vae.decode(z)
            save_image(gen_mask, os.path.join(gen_dir, f'generated_mask_{i}.png'))


if __name__ == "__main__":
    train_vae()