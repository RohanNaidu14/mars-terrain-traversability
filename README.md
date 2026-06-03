# Autonomous Terrain Traversability Analysis for Mars Rovers using Lightweight Semantic Segmentation

> **MSc Dissertation Project — Liverpool John Moores University (LJMU)**
> **Programme:** MSc Data Science and Machine Learning
> **Dataset:** NASA AI4Mars (MSL Curiosity Rover)

---

## Overview

This project develops and evaluates seven lightweight deep learning semantic segmentation architectures for autonomous terrain traversability analysis on Mars. Using the NASA AI4Mars dataset captured by the MSL Curiosity rover, models are trained to identify terrain types and generate binary safe/unsafe traversability maps to support autonomous rover navigation.

The system performs two tasks:

- **Task 1 — Terrain Segmentation:** Classifies every pixel of a Mars surface image into one of four terrain classes: Soil, Bedrock, Sand, Big Rock
- **Task 2 — Traversability Mapping:** Converts terrain predictions into a binary Safe/Unsafe map to guide rover path planning

**Total Dataset Size:** 16,064 labelled image-mask pairs available.
Experiments used 100 (V1), 500 (V2), and 1000 (V3) diverse pairs
selected from this pool using a class-diversity strategy.

---

## Sample Visual Output

| Original Image | GT Terrain Mask | Pred Terrain Mask | GT Safe/Unsafe | Pred Safe/Unsafe |
|---|---|---|---|---|
| Mars EDR Image | Ground Truth | Model Prediction | Ground Truth Map | Predicted Map |

*Five-panel visual output generated for every test image across all 7 models and all 3 experimental runs.*

---

## Dataset — NASA AI4Mars

**Source:** [AI4Mars Dataset — NASA JPL](https://doi.org/10.48550/arXiv.2112.05869)

The AI4Mars dataset provides terrain-labelled images from the MSL (Mars Science Laboratory) Curiosity rover. Labels were generated through crowdsourcing with a minimum 3-labeller agreement and 2/3 majority vote, producing high-quality consensus masks.

### Dataset Structure
```
ai4mars-dataset-merged-0.1/
└── msl/
    ├── images/
    │   └── edr/          ← grayscale EDR JPG images (16,064 available)
    └── labels/
        └── train/        ← PNG segmentation masks (R=G=B encoding)
```

### Terrain Class Encoding

| Pixel Value | Class | Traversability | Reason |
|---|---|---|---|
| 0 | Soil | Safe | Flat, compact — safe for rover wheels |
| 1 | Bedrock | Safe | Solid rock surface — stable for traversal |
| 2 | Sand | Unsafe | Loose, slippery — risk of wheel sinkage |
| 3 | Big Rock | Unsafe | Obstacle — risk of rover damage |
| 255 | NULL | Ignored | Unlabelled pixels — excluded from training and evaluation |

### How Masking Works

Each image-mask pair consists of one Mars EDR grayscale image and one corresponding PNG label mask. The mask encodes terrain classes as pixel integer values (0–3) in the R channel (R=G=B in AI4Mars encoding). **NULL pixels (value=255) are excluded from all loss calculations and metrics** using `ignore_index=255` in the loss function.

**Critically, one single mask can contain all four terrain classes simultaneously** in different spatial regions of the same image. This means the dataset does not require equal numbers of images per class — instead, class diversity is achieved by selecting images whose masks contain rare terrain (Sand, Big Rock) alongside dominant terrain (Soil, Bedrock). This avoids the need to collect separate images per class and directly reflects real Martian surface complexity.

### Class Imbalance and Weighted Loss

Soil dominates the dataset (~45% of valid pixels), while Big Rock is extremely rare (~1.4%). To prevent class collapse:

- **Inverse-frequency class weights** were computed from training mask pixel counts
- **Class weights applied:** Soil=0.11, Bedrock=0.11, Sand=0.43, Big Rock=3.35
- **Combined loss:** Weighted CrossEntropyLoss + DiceLoss

---

## Experimental Design

Three experimental runs were conducted to study the effect of dataset size on model performance:

| Run | Images | Train | Validation | Test |
|---|---|---|---|---|
| V1 | 100 | 70 | 15 | 15 |
| V2 | 500 | 350 | 75 | 75 |
| V3 | 1000 | 700 | 150 | 150 |

### Diverse Image Selection Strategy

Rather than randomly sampling images, a diversity-first selection strategy was applied:
- **~50%** of selected images contain Sand or Big Rock (rare/unsafe classes)
- **~25%** contain Bedrock as the dominant class
- **~25%** are Soil-only images

This ensures rare terrain classes are well-represented without requiring equal per-class image counts, which would be impossible given the natural scarcity of Big Rock in the dataset.

All 7 models within each run use **identical split files** (train.txt, val.txt, test.txt) for a fully fair comparison.

---

## Models Evaluated

| Exp | Model | Architecture | Parameters | Size |
|---|---|---|---|---|
| EXP-01 | U-Net | Scratch, encoder-decoder | 31.4M | 119.7 MB |
| EXP-02 | Mini U-Net | Scratch, halved filters | 7.8M | 29.9 MB |
| EXP-03 | Attention U-Net | Scratch, gated skip connections | 31.9M | 121.7 MB |
| EXP-04 | MobileNetV2-UNet | Pretrained encoder + decoder | 6.6M | 25.3 MB |
| EXP-05 | MobileNetV3-UNet | Pretrained encoder + SE blocks | 6.7M | 25.5 MB |
| EXP-06 | EfficientNet-B0-UNet | Pretrained compound scaling | 8.5M | 32.3 MB |
| EXP-07 | ResNet18-UNet | Pretrained residual encoder | 14.4M | 55.0 MB |

### Architecture Notes

- **Scratch models (EXP-01, 02, 03):** Trained entirely on Mars data — no domain gap
- **Pretrained models (EXP-04 to 07):** ImageNet RGB weights transferred to grayscale Mars imagery — introduces a domain gap that is studied as a key finding
- **Dual skip connections (EXP-07):** ResNet18 has internal residual connections AND external U-Net skip connections
- **SE blocks (EXP-05, 06):** Squeeze-and-Excitation channel attention inside the encoder

---

## Traversability Mapping (Task 2)

A two-stage pipeline is used:

```
Stage 1 (Task 1): Image → [Model] → 4-class terrain mask
Stage 2 (Task 2): Terrain mask → [TRAVERSABILITY_MAP] → Binary Safe/Unsafe map
```

Mapping rule:
```python
TRAVERSABILITY_MAP = {
    0: 0,   # Soil     → Safe
    1: 0,   # Bedrock  → Safe
    2: 1,   # Sand     → Unsafe
    3: 1,   # Big Rock → Unsafe
}
```

---

## Training Configuration

```python
IMAGE_SIZE    = (256, 256)
BATCH_SIZE    = 4           # 8 for Mini U-Net
NUM_EPOCHS    = 50
LEARNING_RATE = 0.001
OPTIMIZER     = Adam
SCHEDULER     = ReduceLROnPlateau (patience=5, factor=0.5)
LOSS          = Weighted CrossEntropyLoss + DiceLoss
DEVICE        = Apple M1 MPS (falls back to CUDA or CPU)
```

### Augmentation Pipeline (Training only)
- Horizontal flip (p=0.5)
- Random rotation ±10° (mask border filled with NULL=255)
- Brightness/contrast adjustment ±20%
- Random resized crop (scale 0.8–1.0)
- All spatial transforms applied identically to image and mask

### Pretrained Model Normalisation
```python
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
```
Grayscale EDR images are converted to 3-channel RGB by replicating the single channel.

---

## Evaluation Metrics

### Task 1 — Segmentation
| Metric | Description |
|---|---|
| Pixel Accuracy | Correctly classified pixels / total valid pixels |
| Mean IoU | Average Intersection over Union across 4 classes |
| Dice Score | Average Dice coefficient across 4 classes |
| Class-wise Recall | Per-class recall (Soil, Bedrock, Sand, Big Rock) |

### Task 2 — Safety (Priority Order)
| Priority | Metric | Description |
|---|---|---|
| 1 | **Unsafe Recall** | Proportion of unsafe pixels correctly identified |
| 2 | **Hazard FNR** | False negative rate for unsafe terrain (missed hazards) |
| 3 | Safe/Unsafe F1 | Harmonic mean of precision and recall |
| 4 | Confusion Matrix | TP, TN, FP, FN for safe/unsafe classification |

> Unsafe Recall and Hazard FNR are the primary safety metrics. Missing unsafe terrain (high FNR) is more dangerous for rover navigation than false alarms (high FPR).

---

## Results Summary

### Best Models by Task and Dataset Size

| Run | Images | Task 1 Best (mIoU) | Task 2 Best (Unsafe Recall) |
|---|---|---|---|
| V1 | 100 | MobileNetV3-UNet (0.2792) | Mini U-Net (0.3562) |
| V2 | 500 | MobileNetV3-UNet (0.2695) | U-Net (0.2913) |
| V3 | 1000 | MobileNetV3-UNet (0.2921) | Mini U-Net (0.4879) |

### V3 Results — 1000 Images (Task 1)

| Model | Pixel Accuracy | Mean IoU | Dice Score | Big Rock Recall |
|---|---|---|---|---|
| U-Net | 0.9017 | 0.2667 | 0.2832 | 0.1172 |
| **Mini U-Net** | 0.9156 | 0.2767 | 0.2962 | **0.1464** |
| Attention U-Net | 0.8912 | 0.2618 | 0.2804 | 0.1228 |
| MobileNetV2-UNet | 0.9136 | 0.2686 | 0.2804 | 0.0660 |
| **MobileNetV3-UNet** | **0.9580** | **0.2921** | **0.3013** | 0.1034 |
| EfficientNet-B0-UNet | 0.9179 | 0.2693 | 0.2802 | 0.0694 |
| ResNet18-UNet | 0.8536 | 0.2408 | 0.2615 | 0.0884 |

### V3 Results — 1000 Images (Task 2)

| Model | Unsafe Recall | Hazard FNR | FPS |
|---|---|---|---|
| U-Net | 0.4506 | 0.1494 | 29.1 |
| **Mini U-Net** | **0.4879** | **0.1121** | **55.3** |
| Attention U-Net | 0.4263 | 0.1737 | 26.8 |
| MobileNetV2-UNet | 0.4096 | 0.1904 | 44.5 |
| MobileNetV3-UNet | 0.3628 | 0.2372 | 52.1 |
| EfficientNet-B0-UNet | 0.3215 | 0.2785 | 46.1 |
| ResNet18-UNet | 0.4136 | 0.1864 | 59.2 |

### Key Findings

1. **MobileNetV3-UNet** is the consistent Task 1 winner across all 3 runs — SE channel attention reliably produces the best terrain segmentation regardless of dataset size
2. **Mini U-Net** is the consistent Task 2 winner — best safety performance at 100 and 1000 images, proving lightweight scratch-trained models outperform heavier pretrained encoders on safety-critical metrics
3. **Domain gap** — pretrained ImageNet RGB encoders underperform scratch models on safety metrics due to the mismatch between colour natural images and grayscale Mars terrain
4. **More data helps safety** — Unsafe Recall improves significantly from v1→v3: Mini U-Net 0.3562 → 0.4879, U-Net 0.3469 → 0.4506
5. **High Pixel Accuracy can be misleading** — MobileNetV3 achieves 0.9580 Pixel Accuracy but has the worst Hazard FNR (0.2372) in v3, driven by dominant Soil class prediction

---

## Project Structure

```
├── data/
│   ├── train.txt               ← image-mask path pairs for training
│   ├── val.txt                 ← image-mask path pairs for validation
│   └── test.txt                ← image-mask path pairs for testing
│
├── notebooks/
│   ├── v1/                     ← 7 notebooks — 100 images
│   │   ├── 01_unet_experiment.ipynb
│   │   ├── 02_mini_unet_experiment.ipynb
│   │   ├── 03_attention_unet_experiment.ipynb
│   │   ├── 04_mobilenetv2_unet_experiment.ipynb
│   │   ├── 05_mobilenetv3_unet_experiment.ipynb
│   │   ├── 06_efficientnet_b0_unet_experiment.ipynb
│   │   └── 07_resnet18_unet_experiment.ipynb
│   ├── v2/                     ← 7 notebooks — 500 images
│   └── v3/                     ← 7 notebooks — 1000 images
│
├── outputs/
│   ├── excel_results/
│   │   ├── v1/                 ← per-model Excel reports (100 images)
│   │   ├── v2/                 ← per-model Excel reports (500 images)
│   │   └── v3/                 ← per-model Excel reports (1000 images)
│   ├── saved_models/           ← best model weights (.pth via Git LFS)
│   │   ├── v1/ v2/ v3/
│   ├── predictions_task1/      ← predicted terrain masks
│   ├── traversability_task2/   ← binary safe/unsafe maps
│   ├── visual_results/         ← 5-panel visual comparison outputs
│   └── logs/                   ← training logs per model per run
│
└── results/
    ├── v1_models_result_sheet.xlsx
    ├── v2_models_result_sheet.xlsx
    └── v3_models_result_sheet.xlsx
```

---

## Excel Output Structure

Each model produces a 6-sheet Excel report:

| Sheet | Contents |
|---|---|
| Training_History | Epoch-wise train/val loss, mIoU, Dice |
| Task1_Image_Wise_Results | Per-image segmentation metrics |
| Task2_Image_Wise_Results | Per-image safety metrics + confusion matrix |
| Task1_Overall_Summary | Average Task 1 metrics, model size, FPS |
| Task2_Overall_Summary | Average Task 2 metrics, total TP/TN/FP/FN |
| Final_Model_Conclusion | Strengths, weaknesses, final verdict |

---

## How to Run

### Prerequisites
```bash
pip install torch torchvision numpy pandas pillow opencv-python matplotlib openpyxl
```

### Dataset Setup
Download the AI4Mars dataset and place it at:
```
/path/to/Dataset/ai4mars-dataset-merged-0.1/
```
Update `DATA_ROOT` in Section 2 of each notebook to match your local path.

### Running the Experiments

**Run Notebook 1 first** for each version — it scans the dataset, selects diverse image-mask pairs, and generates the shared split files used by all 7 notebooks.

```
notebooks/v1/01_unet_experiment.ipynb    ← Run first (generates split files)
notebooks/v1/02_mini_unet_experiment.ipynb
...
notebooks/v1/07_resnet18_unet_experiment.ipynb
```

Each notebook runs top to bottom and produces:
- Trained model weights (.pth)
- 6-sheet Excel results report
- Predicted terrain masks per test image
- Binary traversability maps per test image
- 5-panel visual comparison images
- Training log file

### Version Configuration (Section 2 of each notebook)

| Setting | V1 | V2 | V3 |
|---|---|---|---|
| NUM_IMAGES | 100 | 500 | 1000 |
| TRAIN_COUNT | 70 | 350 | 700 |
| VAL_COUNT | 15 | 75 | 150 |
| TEST_COUNT | 15 | 75 | 150 |
| PROJECT_ROOT | `Path("../")` | `Path("../../")` | `Path("../../")` |
| VERSION | — | `"v2"` | `"v3"` |

---

## Hardware

Trained on **Apple M1 MacBook Pro** using MPS (Metal Performance Shaders) backend.
Automatically falls back to CUDA (NVIDIA GPU) or CPU if MPS is unavailable.

> Note: Apple M1 is used as a simulation and training platform. Inference times reported are relative comparisons between models and do not represent absolute real-world rover deployment performance.

---

## Priority-Based Evaluation Criteria

| Priority | Metric | Why Important |
|---|---|---|
| 1 | Unsafe Recall | Model must detect all unsafe sand and rock regions |
| 2 | Hazard FNR | Missing unsafe terrain can strand or damage the rover |
| 3 | Mean IoU | Overall terrain segmentation quality |
| 4 | Dice Score | Mask similarity — useful for small datasets |
| 5 | Class-wise Recall | Detection quality per terrain class |
| 6 | Inference FPS | Suitability for real-time rover navigation |
| 7 | Model Size | Deployment on resource-constrained onboard hardware |

---

## Acknowledgements

- **NASA AI4Mars Dataset** — JPL/Caltech
- **Liverpool John Moores University** — MSc Data Science and Machine Learning
- **PyTorch** — Deep learning framework
- **torchvision** — Pretrained model weights (ImageNet)

---

## License

This project is for academic research purposes only.

---
