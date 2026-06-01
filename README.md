# Autonomous Terrain Traversability Analysis for Mars Rovers using Lightweight Semantic Segmentation

> MSc Dissertation Project — Liverpool John Moores University (LJMU)  
> Dataset: NASA AI4Mars (MSL Curiosity Rover)

---

## Overview

This project develops and evaluates lightweight deep learning models for autonomous terrain traversability analysis on Mars. Using the NASA AI4Mars dataset captured by the MSL Curiosity rover, seven semantic segmentation architectures are trained and compared across two experimental runs (v1: 100 images, v2: 500 images).

The system performs two tasks:
- **Task 1:** Multi-class terrain segmentation — Soil, Bedrock, Sand, Big Rock
- **Task 2:** Binary safe/unsafe traversability mapping for rover navigation

---

## Sample Input and Label

| Input Image (EDR) | Terrain Label Mask |
|---|---|
| ![Sample Mars EDR Image](notebooks/sample_input.jpg) | ![Sample Label Mask](notebooks/sample_label.png) |

*Left: Raw grayscale EDR image from NASA MSL Curiosity rover. Right: Corresponding terrain label mask (white = NULL/unlabelled region, black = labelled terrain).*

---

## Models Evaluated

| Exp | Model | Parameters | Size |
|---|---|---|---|
| EXP-01 | U-Net (Baseline) | 31.4M | 119.7 MB |
| EXP-02 | Mini U-Net | 7.8M | 29.9 MB |
| EXP-03 | Attention U-Net | 31.9M | 121.7 MB |
| EXP-04 | MobileNetV2-UNet | 6.6M | 25.3 MB |
| EXP-05 | MobileNetV3-UNet | 6.7M | 25.5 MB |
| EXP-06 | EfficientNet-B0-UNet | 8.5M | 32.3 MB |
| EXP-07 | ResNet18-UNet | 14.4M | 55.0 MB |

---

## Dataset

**NASA AI4Mars** — Terrain-labelled images from the MSL Curiosity rover.

- Images: Grayscale EDR JPG format
- Labels: 4-class PNG masks (R=G=B encoding)
  - `0` = Soil
  - `1` = Bedrock
  - `2` = Sand
  - `3` = Big Rock
  - `255` = NULL (unlabelled)

Download the dataset from: [NASA AI4Mars on Kaggle](https://www.kaggle.com/datasets/NASA AI4Mars)

Place the dataset at:
```
Dataset/ai4mars-dataset-merged-0.1/
├── msl/
│   ├── images/edr/        ← input images (.JPG)
│   └── labels/train/      ← label masks (.png)
```

---

## Project Structure

```
├── data/
│   ├── train.txt          ← training split (image-mask path pairs)
│   ├── val.txt            ← validation split
│   └── test.txt           ← test split
│
├── notebooks/
│   ├── v1/                ← 7 notebooks trained on 100 images
│   │   ├── 01_unet_experiment.ipynb
│   │   ├── 02_mini_unet_experiment.ipynb
│   │   ├── 03_attention_unet_experiment.ipynb
│   │   ├── 04_mobilenetv2_unet_experiment.ipynb
│   │   ├── 05_mobilenetv3_unet_experiment.ipynb
│   │   ├── 06_efficientnet_b0_unet_experiment.ipynb
│   │   └── 07_resnet18_unet_experiment.ipynb
│   └── v2/                ← 7 notebooks trained on 500 images
│
├── outputs/
│   ├── excel_results/     ← per-model Excel reports (v1 and v2)
│   ├── saved_models/      ← best model weights (.pth via Git LFS)
│   ├── predictions_task1/ ← predicted terrain masks
│   ├── traversability_task2/ ← safe/unsafe maps
│   ├── visual_results/    ← 5-panel visual comparison outputs
│   └── logs/              ← training logs
│
└── results/
    ├── v1_models_result_sheet.xlsx
    └── v2_models_result_sheet.xlsx
```

---

## Results Summary (V1 — 100 Images)

### Task 1 — Terrain Segmentation

| Model | Mean IoU | Dice Score | Pixel Accuracy |
|---|---|---|---|
| U-Net | 0.2002 | 0.2361 | 0.6100 |
| Mini U-Net | 0.2236 | 0.2627 | 0.6814 |
| Attention U-Net | 0.1947 | 0.2399 | 0.5999 |
| MobileNetV2-UNet | 0.2252 | 0.2403 | 0.8318 |
| MobileNetV3-UNet | **0.2792** | **0.2915** | **0.8853** |
| EfficientNet-B0-UNet | 0.2684 | **0.2940** | 0.8447 |
| ResNet18-UNet | 0.2124 | 0.2442 | 0.7135 |

### Task 2 — Traversability Safety

| Model | Unsafe Recall | Hazard FNR | FPS |
|---|---|---|---|
| U-Net | 0.3469 | 0.0531 | 29.8 |
| **Mini U-Net** | **0.3562** | **0.0438** | **55.5** |
| Attention U-Net | 0.3135 | 0.0865 | 26.2 |
| MobileNetV2-UNet | 0.2526 | 0.1474 | 51.1 |
| MobileNetV3-UNet | 0.2400 | 0.1600 | 46.7 |
| EfficientNet-B0-UNet | 0.3280 | 0.0720 | 40.6 |
| ResNet18-UNet | 0.2100 | 0.1900 | 51.6 |

**Overall Rank 1: Mini U-Net** — Best Unsafe Recall (0.3562), lowest Hazard FNR (0.0438), fastest inference among scratch models.

---

## Traversability Mapping

Terrain classes are mapped to binary traversability:

| Terrain Class | Traversability |
|---|---|
| Soil | Safe (0) |
| Bedrock | Safe (0) |
| Sand | Unsafe (1) |
| Big Rock | Unsafe (1) |

---

## Priority-Based Evaluation Criteria

Ranking is based on the following priority order:

1. **Unsafe Recall** — Must detect sand and rock hazards
2. **Hazard FNR** — Missing unsafe terrain is dangerous for rover safety
3. **Mean IoU** — Overall segmentation quality
4. **Dice Score** — Mask similarity
5. **Class-wise Recall** — Detection of all 4 terrain types
6. **Inference Time / FPS** — Real-time deployment suitability
7. **Model Size** — Onboard hardware constraints

---

## How to Run

### Prerequisites

```bash
pip install torch torchvision numpy pandas pillow opencv-python matplotlib openpyxl
```

### Running a Notebook

1. Update `DATA_ROOT` in Section 2 to point to your local AI4Mars dataset path.
2. Run Notebook 1 first — it generates the shared `train.txt`, `val.txt`, `test.txt` split files used by all other notebooks.
3. Run notebooks 2–7 in any order.

```
notebooks/v1/01_unet_experiment.ipynb        ← Run first (generates split files)
notebooks/v1/02_mini_unet_experiment.ipynb
...
notebooks/v1/07_resnet18_unet_experiment.ipynb
```

Each notebook produces:
- Trained model weights (`.pth`)
- Excel report with 6 sheets of results
- Predicted terrain masks
- Safe/unsafe traversability maps
- 5-panel visual comparison images
- Training log

---

## Hardware

Trained on Apple M1 (MPS backend). Automatically falls back to CUDA or CPU.

---

## Acknowledgements

- NASA AI4Mars dataset — JPL/Caltech
- Liverpool John Moores University — MSc Data Science and Machine Learning

---

## License

This project is for academic research purposes only.
