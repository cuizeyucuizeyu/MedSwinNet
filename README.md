# Swin Transformer with Gated Multi-Scale Fusion for Paired Image Classification

This repository contains the training code for a customized Swin Transformer framework for binary image classification with a paired-image evaluation strategy.

The model is based on a Swin Transformer backbone and includes:
- multi-stage feature extraction,
- gated multi-scale feature fusion,
- CBAM-based attention refinement,
- severe-side-only training,
- paired left/right validation and testing.
---

## 1. Repository Structure

```text
.
├── train.py
├── model.py
├── utils.py
├── requirements.txt
├── README.md
└── .gitignore
````

---

## 2. Environment

Recommended:

* Python 3.10 or 3.11

### Step 1: Install PyTorch

Choose one of the following commands depending on your environment.

#### CPU only

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cpu
```

#### CUDA 12.1

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

#### CUDA 11.8

```bash
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu118
```

### Step 2: Install the remaining dependencies

```bash
pip install -r requirements.txt
```

---

## 3. Requirements

The `requirements.txt` file should contain:

```txt
numpy==1.26.4
scikit-learn==1.5.2
Pillow==10.4.0
matplotlib==3.9.2
tensorboard==2.18.0
tqdm>=4.66.0
```

---

## 4. Dataset Format

The dataset should follow the `ImageFolder` directory structure:

```text
dataset/
├── class0/
│   ├── case001_left_l.jpg
│   ├── case001_left_r.jpg
│   ├── case002_right_l.jpg
│   ├── case002_right_r.jpg
│   └── ...
└── class1/
    ├── case101_left_l.jpg
    ├── case101_left_r.jpg
    ├── case102_right_l.jpg
    ├── case102_right_r.jpg
    └── ...
```

### Filename Convention

This code relies on the filename pattern to:

1. group images by case,
2. identify the severe side,
3. construct left/right image pairs.

The filenames must satisfy the following rules:

* contain `_left_` or `_right_`
* end with `_l.jpg` or `_r.jpg`

Example:

* `case001_left_l.jpg`
* `case001_left_r.jpg`

Interpretation:

* `case001` is the case prefix
* `left` indicates the severe side
* `_l` and `_r` indicate left-image / right-image endings used by the pairing logic

If the naming pattern is inconsistent, pair construction may fail.

---

## 5. Model Overview

The customized model defined in `model3_5.py` contains:

* a Swin Transformer backbone,
* multi-stage feature extraction,
* gated weighting across multiple scales,
* CBAM attention after feature concatenation,
* a 1×1 convolution fusion head,
* global average pooling,
* a final classification head.

The multi-scale fusion process works as follows:

1. Extract features from all Swin stages.
2. Resize earlier-stage features to the final-stage resolution.
3. Apply learnable gate weights to each scale.
4. Concatenate all weighted features.
5. Refine the fused feature with CBAM.
6. Project fused channels with a 1×1 convolution before classification.

---

## 6. Training Strategy

The training procedure in `train.py` is:

1. Load the full dataset using `torchvision.datasets.ImageFolder`
2. Group samples by case prefix to prevent left/right leakage across splits
3. Split grouped cases into:

   * a train/validation subset
   * an independent held-out test subset
4. Apply K-fold cross-validation on the train/validation subset
5. Use only severe-side images for training
6. Use paired left/right images for validation and testing

### Data Transforms

The code uses:

* random augmentation for training
* deterministic transforms for validation and testing

This design improves reproducibility and prevents random augmentation from affecting evaluation results.

---

## 7. Training

Example command:

```bash
python train.py \
  --seed 27 \
  --num_classes 2 \
  --epochs 120 \
  --batch-size 8 \
  --lr 1e-4 \
  --data-path ./dataset \
  --weights ./swin_base_patch4_window7_224_22k.pth \
  --save-dir ./weights_seed27 \
  --log-dir ./runs \
  --device cuda:0
```

Train without pretrained weights:

```bash
python train.py --weights ""
```

Freeze the backbone and train only the classification head:

```bash
python train.py --freeze-layers
```

---

## 8. Output Files

During training, the script saves:

* the best model checkpoint for each fold in `save_dir`
* TensorBoard logs in `log_dir`
* `config.json`
* `results_summary.json`

---

## 9. TensorBoard

To monitor training logs:

```bash
tensorboard --logdir ./runs
```

Then open the local URL shown in the terminal.

---

## 10. Evaluation

The paired evaluation function in `utils1.py` provides:

* single-side accuracy
* double-side accuracy

By default:

* single-side prediction is based on the left image branch
* double-side prediction follows the original implementation logic

If you later want a different paired fusion strategy, you can modify `evaluate_pairs()` accordingly.

---

## 11. Pretrained Weights

If you use pretrained Swin weights, place the `.pth` file in the project root, for example:

```text
swin_base_patch4_window7_224_22k.pth
```

Then specify it with:

```bash
--weights ./swin_base_patch4_window7_224_22k.pth
```

If you do not want to use pretrained weights:

```bash
--weights ""
```

---

## 12. Reproducibility

The training script includes random seed control for:

* Python `random`
* NumPy
* PyTorch

This improves reproducibility, although exact results may still vary slightly across different hardware and software environments.

---

## 13. Notes

* Make sure your dataset filenames strictly follow the expected naming rules.
* The current training logic assumes binary classification.
* The current evaluation logic preserves the original project behavior for paired prediction.

---

## 14. Citation

If you use this repository in your work, please cite the corresponding paper or acknowledge this repository.

---




