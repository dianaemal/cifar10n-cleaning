"""Step 2: embed all CIFAR-10 images with CLIP ViT-B/32 and save L2-normalized features.
Run on a GPU (Colab or lab PC)."""
import os

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader

from config import DATA_DIR, FEAT_DIR

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
model = model.to(device).eval()


def embed(train_flag):
    ds = torchvision.datasets.CIFAR10(DATA_DIR, train=train_flag, download=True,
                                      transform=preprocess)
    loader = DataLoader(ds, batch_size=256, num_workers=2, shuffle=False)  # keep order!
    out = []
    with torch.no_grad():
        for i, (x, _) in enumerate(loader):
            f = model.encode_image(x.to(device))
            out.append(F.normalize(f, dim=-1).float().cpu())
            if i % 20 == 0:
                print(f"  batch {i}/{len(loader)}")
    return torch.cat(out).numpy()


X_train = embed(True)
np.save(os.path.join(FEAT_DIR, "X_train.npy"), X_train)
X_test = embed(False)
np.save(os.path.join(FEAT_DIR, "X_test.npy"), X_test)

print("X_train", X_train.shape, "X_test", X_test.shape)
print("Norm check (should be 1.0):", np.linalg.norm(X_train[:5], axis=1))
