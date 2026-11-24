# Finding and Fixing Mislabeled Samples in CIFAR-10N

Topic 1: Enhancing Quality in Crowdsourced Datasets.

We compare two methods for detecting mislabeled images in CIFAR-10N
(logistic-regression disagreement vs. k-nearest-neighbour label consistency),
suggest corrected labels, measure detection precision/recall, and measure
whether cleaning improves test accuracy. A Streamlit tool lets a reviewer
accept or reject each suggestion.

## Label version
We use `random_label1` (single annotator, ~17% noise) from CIFAR-10N.

## Pipeline
| Step | Script | Output |
|---|---|---|
| 1. Load labels | `src/01_load_labels.py` | `features/y_noisy.npy`, `y_clean.npy`, `y_test.npy` |
| 2. CLIP embeddings (GPU) | `src/02_embed.py` or `notebooks/colab_embed.ipynb` | `features/X_train.npy`, `X_test.npy` |
| 3. Baseline | `src/03_baseline.py` | accuracy A, oracle accuracy |
| 4. Detection | `src/04_detect.py` | flags, suggested labels |
| 5. Evaluation | `src/05_evaluate.py` | tables and plots in `results/` |
| 6. Review tool | `app.py` | `results/decisions.csv` |

## Setup
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Steps 1–2 are easiest on Google Colab (free GPU): open
`notebooks/colab_embed.ipynb`, run all cells, then download the `features/`
folder from Google Drive into this repo's `features/`. Steps 3+ run on a laptop.

## Paths
Scripts read the project root from the `PROJECT_DIR` environment variable
(defaults to the current directory). Data and features are gitignored.
