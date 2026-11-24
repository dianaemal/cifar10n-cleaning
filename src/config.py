"""Shared paths and settings. Every script imports from here so settings stay identical."""
import os

ROOT = os.environ.get("PROJECT_DIR", ".")
DATA_DIR = os.path.join(ROOT, "data")
FEAT_DIR = os.path.join(ROOT, "features")
RESULTS_DIR = os.path.join(ROOT, "results")

LABEL_VERSION = "random_label1"   # state this in the report
SEED = 42

# Fixed BEFORE looking at clean labels (do not tune on clean labels)
LR_PARAMS = dict(max_iter=2000, C=1.0)
CONF_THRESH = 0.90    # Method A
K = 10                # Method B
MIN_VOTES = 6         # Method B: "most" neighbours = at least 6 of 10

CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]

for d in (DATA_DIR, FEAT_DIR, RESULTS_DIR):
    os.makedirs(d, exist_ok=True)
