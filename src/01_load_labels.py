"""Step 1: load CIFAR-10 + CIFAR-10N labels, verify order, save label arrays."""
import os
import urllib.request

import numpy as np
import torch
import torchvision

from config import DATA_DIR, FEAT_DIR, LABEL_VERSION

HUMAN_URL = "https://github.com/UCSC-REAL/cifar-10-100n/raw/main/data/CIFAR-10_human.pt"
human_path = os.path.join(DATA_DIR, "CIFAR-10_human.pt")
if not os.path.exists(human_path):
    print("Downloading CIFAR-10N labels...")
    urllib.request.urlretrieve(HUMAN_URL, human_path)

train = torchvision.datasets.CIFAR10(DATA_DIR, train=True, download=True)
test = torchvision.datasets.CIFAR10(DATA_DIR, train=False, download=True)

human = torch.load(human_path, weights_only=False)
print("Label versions in file:", list(human.keys()))

y_clean = np.array(train.targets)
# Critical check: CIFAR-10N labels must be in the same order as torchvision images
assert (np.array(human["clean_label"]) == y_clean).all(), "Label order mismatch!"

y_noisy = np.array(human[LABEL_VERSION])
y_test = np.array(test.targets)

print(f"Using {LABEL_VERSION}: noise rate = {(y_noisy != y_clean).mean():.4f}")
for k in human:
    if k != "clean_label":
        print(f"  {k:15s} noise rate = {(np.array(human[k]) != y_clean).mean():.4f}")

np.save(os.path.join(FEAT_DIR, "y_noisy.npy"), y_noisy)
np.save(os.path.join(FEAT_DIR, "y_clean.npy"), y_clean)
np.save(os.path.join(FEAT_DIR, "y_test.npy"), y_test)
print("Saved labels to", FEAT_DIR)
