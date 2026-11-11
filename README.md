# COVID-19 CT Scan Classification with MobileNetV3

A PyTorch notebook that classifies chest CT slices into three categories using transfer learning on a pretrained **MobileNetV3-Large**. It is built to run end-to-end in **Google Colab**, with automatic fallback to a fast "quick-test" mode when no GPU is available.

> ⚠️ **Disclaimer:** This is an educational / research project. It is **not** a medical device and must not be used for clinical diagnosis.

---

## Table of Contents

- [Overview](#overview)
- [Classes](#classes)
- [Dataset](#dataset)
- [Model & Approach](#model--approach)
- [Results](#results)
- [Getting Started](#getting-started)
- [Configuration](#configuration)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Using the Saved Model](#using-the-saved-model)
- [Known Issues & Limitations](#known-issues--limitations)
- [Future Improvements](#future-improvements)

---

## Overview

| | |
|---|---|
| **Task** | 3-class image classification of CT slices |
| **Framework** | PyTorch + torchvision |
| **Backbone** | MobileNetV3-Large (ImageNet pretrained) |
| **Input size** | 224 × 224 RGB |
| **Platform** | Google Colab (GPU recommended) |
| **Notebook** | `Covid19_Scans.ipynb` |

## Classes

| Folder | Meaning |
|---|---|
| `nCT` | Normal CT |
| `pCT` | COVID-19 positive CT |
| `NiCT` | Non-informative CT (slices without useful diagnostic content) |

## Dataset

The notebook expects a `archive.zip` in your Google Drive containing a `Preprocessed CT scans/` folder with one sub-folder per class:

```
Preprocessed CT scans/
├── NiCT/    # 5,705 images
├── nCT/     # 9,979 images
└── pCT/     # 4,001 images
```

**Total: 19,685 images.** The classes are imbalanced (normal slices make up roughly half the data), which is worth keeping in mind when reading accuracy numbers.

The archive also contains an `Original CT Scans` folder, which this notebook does not use.

> Add the dataset source / link and license here.

## Model & Approach

1. **Data loading:** `torchvision.datasets.ImageFolder` reads the class folders.
2. **Preprocessing:** images are normalized with ImageNet mean/std.
   - *Training:* `Resize(256)` → `RandomResizedCrop(224)` → `RandomHorizontalFlip` → normalize
   - *Evaluation:* `Resize(256)` → `CenterCrop(224)` → normalize
3. **Split:** stratified per class, **70% train / 15% validation / 15% test** (seeded for reproducibility).
4. **Model:** MobileNetV3-Large with the classifier head replaced by:
   `Linear(960→1280) → ReLU → Dropout(0.2) → Linear(1280→3)`
5. **Training:** `AdamW` (lr `1e-4`, weight decay `1e-4`), `CrossEntropyLoss`, batch size 32.
6. **Model selection:** the weights from the epoch with the best **validation accuracy** are restored at the end of training.
7. **Evaluation:** test accuracy, confusion matrix, per-class precision and recall.
8. **Inference:** a helper that shows a CT slice with its predicted class and probability bars.

### Two run modes

The notebook picks a mode automatically based on whether a GPU is detected:

| Setting | Quick-test (CPU) | Full (GPU) |
|---|---|---|
| Images per class | 300 | all |
| Backbone | frozen (only classifier trains) | fully fine-tuned |
| Copy data to local disk | no | yes |
| Epochs | 100 | 100 |

## Results

The figures below come from the **quick-test run on CPU** (900 images total: 630 train / 135 val / 135 test, backbone frozen, 100 epochs, ~93 minutes).

| Metric | Value |
|---|---|
| Trainable parameters | 1,233,923 of 4,205,875 |
| Best validation accuracy | **92.59%** |
| Test accuracy | **≈ 86.67%** (117 / 135) |

**Per-class test performance**

| Class | Precision | Recall |
|---|---|---|
| NiCT | 93.02% | 88.89% |
| nCT (Normal) | 82.00% | 91.11% |
| pCT (COVID-19) | 85.71% | 80.00% |

> The test accuracy is derived from the per-class precision/recall table in the notebook output (the original accuracy cell output was not saved). COVID-positive slices had the lowest recall, meaning some positive cases were missed. For a screening-type task this is the metric to improve first.

Running in full GPU mode (all ~19.7k images, fine-tuned backbone) should give a more representative result.

## Getting Started

### Prerequisites

- A Google account with Google Drive
- Google Colab (free tier works; a **T4 GPU** is strongly recommended)
- The dataset zip uploaded to your Drive as `archive.zip`

### Run it

1. Upload `archive.zip` to your **My Drive** (root or any sub-folder).
2. Open `Covid19_Scans.ipynb` in Google Colab.
3. *(Recommended)* Enable a GPU: **Runtime → Change runtime type → T4 GPU**.
4. Run all cells in order. The notebook will:
   - mount Drive,
   - locate and extract `archive.zip` to `/content/ct_scans`,
   - train, evaluate, and save the model to `MyDrive/ct_classifier.pth`.

### Dependencies

All of these are preinstalled in Colab:

```
torch, torchvision, numpy, matplotlib, pillow, tqdm
```

To run locally instead, install them with:

```bash
pip install torch torchvision numpy matplotlib pillow tqdm
```

and remove the Google Drive mounting cell, updating `DATA_ROOT` and `SAVE_PATH` to local paths.

## Configuration

All key settings live in one cell near the top:

| Variable | Default | Description |
|---|---|---|
| `DATA_ROOT` | `/content/ct_scans/Preprocessed CT scans` | Folder containing the class sub-folders |
| `SAVE_PATH` | `/content/drive/MyDrive/ct_classifier.pth` | Where the trained model is saved |
| `QUICK_TEST` | `True` if no GPU | Override manually with `True` / `False` |
| `IMG_SIZE` | `224` | Input resolution |
| `BATCH_SIZE` | `32` | Batch size |
| `LEARNING_RATE` | `1e-4` | AdamW learning rate |
| `EPOCHS` | `100` | Training epochs |
| `MAX_IMAGES_PER_CLASS` | `300` (quick) / `None` (full) | Cap on images per class |
| `FREEZE_BACKBONE` | `True` (quick) / `False` (full) | Train only the new head or the whole network |
| `SEED` | `42` | Reproducible shuffling and initialization |

## Notebook Walkthrough

1. **Setup:** mount Drive, imports, device detection
2. **Config:** paths and training settings
3. **Data extraction:** unzip `archive.zip` from Drive
4. **Transforms & datasets:** training augmentation and evaluation preprocessing
5. **Exploration:** class counts bar chart and random sample images per class
6. **Splitting & loaders:** stratified 70/15/15 split and `DataLoader`s
7. **Model:** MobileNetV3-Large with a custom 3-class head
8. **Training loop:** per-epoch train/val metrics, best-weights tracking
9. **Plots:** loss and accuracy curves
10. **Evaluation:** test accuracy, confusion matrix, precision/recall
11. **Save / load:** checkpoint with weights, class names, and image size
12. **Prediction demo:** visualize the prediction for a random test image

## Using the Saved Model

The checkpoint is a dictionary containing `state_dict`, `class_names`, and `img_size`.

```python
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt = torch.load("ct_classifier.pth", map_location=device)

# Rebuild the same architecture
model = models.mobilenet_v3_large(weights=None)
model.classifier = nn.Sequential(
    nn.Linear(960, 1280), nn.ReLU(), nn.Dropout(0.2),
    nn.Linear(1280, len(ckpt["class_names"])),
)
model.load_state_dict(ckpt["state_dict"])
model.to(device).eval()

# Preprocess exactly like evaluation
tfm = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(ckpt["img_size"]),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

img = tfm(Image.open("scan.png").convert("RGB")).unsqueeze(0).to(device)
with torch.no_grad():
    probs = torch.softmax(model(img), dim=1)[0]

print(ckpt["class_names"][probs.argmax()], probs.max().item())
```

## Known Issues & Limitations

- **Small-sample results:** reported metrics come from a 135-image test set, so each image shifts per-class numbers by several percentage points.
- **Class imbalance:** the full dataset is imbalanced; no class weighting or resampling is applied.
- **Slice-level split:** images are split at the slice level. If slices from the same patient land in different splits, scores can be optimistic. Patient-level splitting is preferable if patient IDs are available.
- **`DataLoader` warnings:** the training log contains `AssertionError: can only test a child process` messages from `num_workers=2` worker shutdown. They are harmless, and setting `num_workers=0` removes them.
- **Unused cells:** `COPY_TO_LOCAL` and the extraction step overlap. The extraction step already writes to local disk, so the copy step is skipped in practice. Run the unzip cell before the data-loading cells.
- **`torch._dynamo` stub:** a workaround cell mocks `torch._dynamo` to avoid a circular import error when creating the optimizer. It may be unnecessary on other PyTorch versions.
- **Not for clinical use.**

## Future Improvements

- Run the full-data GPU configuration and report final test metrics
- Add early stopping (100 epochs is more than the validation curve needs)
- Use class weights or a weighted sampler for imbalance
- Add a learning-rate scheduler
- Report F1, ROC-AUC, and sensitivity/specificity for the COVID class
- Add Grad-CAM visualizations for interpretability
- Split by patient to prevent data leakage
- Compare against other backbones (EfficientNet, ResNet, DenseNet)

## License

Add your license here (e.g., MIT) and credit the dataset's original authors.
