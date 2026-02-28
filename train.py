import os
import re
import json
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image
from sklearn.model_selection import KFold, train_test_split

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.datasets import ImageFolder

from utils import evaluate_pairs, train_one_epoch
from model import swin_base_patch4_window7_224_in22k as create_model


# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -----------------------------
# Filename / sample utilities
# -----------------------------
def group_samples_by_prefix(samples: List[Tuple[str, int]]) -> List[List[Tuple[str, int]]]:
    """
    Group samples by case prefix to avoid splitting left/right images across subsets.

    Expected filename pattern contains:
        *_left_* or *_right_*

    Example:
        case001_left_l.jpg
        case001_left_r.jpg
        case002_right_l.jpg
        case002_right_r.jpg
    """
    grouped_samples: Dict[str, List[Tuple[str, int]]] = {}
    pattern = re.compile(r"^(.*?)(_left_|_right_)")

    for path, label in samples:
        filename = os.path.basename(path)
        match = pattern.match(filename)
        if not match:
            continue

        prefix = match.group(1)
        grouped_samples.setdefault(prefix, []).append((path, label))

    return list(grouped_samples.values())


def build_pairs(samples: List[Tuple[str, int]]) -> List[Tuple[str, str, int, str]]:
    """
    Build paired samples from image list.

    Returns:
        List of tuples:
        (left_image_path, right_image_path, label, severe_side)

    severe_side is inferred from filename containing '_left_' or '_right_'.
    """
    pairs = []
    temp_dict = {}
    failed_pairs = []

    for path, label in samples:
        filename = os.path.basename(path)

        try:
            if filename.endswith("_l.jpg"):
                key = filename[:-6]
                side = "l"
            elif filename.endswith("_r.jpg"):
                key = filename[:-6]
                side = "r"
            else:
                continue

            if key not in temp_dict:
                if "_left_" in filename:
                    severe_side = "left"
                elif "_right_" in filename:
                    severe_side = "right"
                else:
                    continue
                temp_dict[key] = {"severe_side": severe_side}

            temp_dict[key][side] = (path, label)

        except Exception as e:
            print(f"[Warning] Failed to process file '{filename}': {e}")

    for key, value in temp_dict.items():
        if "l" in value and "r" in value:
            left = value["l"]
            right = value["r"]
            severe_side = value["severe_side"]
            pairs.append((left[0], right[0], left[1], severe_side))
        else:
            failed_pairs.append(key)

    if failed_pairs:
        print(f"[Warning] Failed to find complete pairs for keys: {failed_pairs}")

    print(f"Total valid pairs: {len(pairs)}")
    return pairs


def is_severe_side_image(path: str) -> bool:
    """
    Keep only the severe-side image according to filename convention:
    - '_left_' paired with '_l.jpg'
    - '_right_' paired with '_r.jpg'
    """
    filename = os.path.basename(path)
    return (
        ("_left_" in filename and filename.endswith("_l.jpg")) or
        ("_right_" in filename and filename.endswith("_r.jpg"))
    )


def build_severe_side_samples(samples: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """Filter training samples to keep only severe-side images."""
    severe_samples = []
    failed_images = []

    for path, label in samples:
        try:
            if is_severe_side_image(path):
                severe_samples.append((path, label))
        except Exception as e:
            print(f"[Warning] Failed to inspect image '{path}': {e}")
            failed_images.append(path)

    if failed_images:
        print(f"[Warning] Failed images: {failed_images}")

    print(f"Total severe side samples: {len(severe_samples)}")
    return severe_samples


# -----------------------------
# Dataset classes
# -----------------------------
class PairedDataset(Dataset):
    """
    Dataset for paired validation/test data.

    Each item returns:
        left_image, right_image, label, severe_side
    """
    def __init__(self, pairs: List[Tuple[str, str, int, str]], transform=None):
        self.pairs = pairs
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        left_path, right_path, label, severe_side = self.pairs[idx]

        try:
            left_image = Image.open(left_path).convert("RGB")
            right_image = Image.open(right_path).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to load paired images:\n  {left_path}\n  {right_path}\nReason: {e}")
            raise

        if self.transform is not None:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)

        return left_image, right_image, label, severe_side


class SevereSideDataset(Dataset):
    """
    Dataset for training data using only severe-side images.

    Each item returns:
        image, label
    """
    def __init__(self, samples: List[Tuple[str, int]], transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"[Error] Failed to load image '{path}': {e}")
            raise

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# -----------------------------
# Transform utilities
# -----------------------------
def build_transforms(img_size: int = 224):
    """
    Create train/eval transforms.

    Important:
    - train uses random augmentation
    - val/test should use deterministic transforms only
    """
    train_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    eval_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.2)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])

    return train_transform, eval_transform


# -----------------------------
# I/O helpers
# -----------------------------
def save_json(obj, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def get_dataloader_kwargs(args, device: torch.device, shuffle: bool):
    kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
    return kwargs


# -----------------------------
# Main training logic
# -----------------------------
def main(args):
    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    writer = SummaryWriter(log_dir=str(log_dir / f"seed_{args.seed}"))

    train_transform, eval_transform = build_transforms(img_size=args.img_size)

    # Load dataset metadata only; actual transform is applied later in custom datasets.
    dataset = ImageFolder(root=args.data_path, transform=None)
    print(f"Total number of images in dataset: {len(dataset)}")

    grouped_samples = group_samples_by_prefix(dataset.samples)
    if len(grouped_samples) == 0:
        raise RuntimeError(
            "No valid grouped samples found. Please check filename pattern "
            "and ensure filenames contain '_left_' or '_right_'."
        )

    indices = np.arange(len(grouped_samples))
    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.seed,
        shuffle=True,
    )

    train_val_samples = [sample for i in train_val_indices for sample in grouped_samples[i]]
    test_samples = [sample for i in test_indices for sample in grouped_samples[i]]

    print(f"Number of training/validation images: {len(train_val_samples)}")
    print(f"Number of test images: {len(test_samples)}")

    # Build independent test set
    test_pairs = build_pairs(test_samples)
    test_dataset = PairedDataset(test_pairs, transform=eval_transform)
    test_loader = DataLoader(
        test_dataset,
        **get_dataloader_kwargs(args, device, shuffle=False)
    )

    # Prepare K-fold on grouped training/validation samples
    grouped_train_val_samples = group_samples_by_prefix(train_val_samples)
    train_val_group_indices = np.arange(len(grouped_train_val_samples))

    if len(grouped_train_val_samples) < args.n_splits:
        raise ValueError(
            f"Number of train/val groups ({len(grouped_train_val_samples)}) "
            f"is smaller than n_splits ({args.n_splits})."
        )

    kfold = KFold(
        n_splits=args.n_splits,
        shuffle=True,
        random_state=args.seed,
    )

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(train_val_group_indices), start=1):
        print(f"\n{'=' * 20} Fold {fold} {'=' * 20}")

        train_groups = [grouped_train_val_samples[i] for i in train_idx]
        val_groups = [grouped_train_val_samples[i] for i in val_idx]

        train_samples = [sample for group in train_groups for sample in group]
        val_samples = [sample for group in val_groups for sample in group]

        # Training set: severe-side only
        train_severe_samples = build_severe_side_samples(train_samples)
        train_dataset = SevereSideDataset(train_severe_samples, transform=train_transform)

        # Validation set: paired left/right images
        val_pairs = build_pairs(val_samples)
        val_dataset = PairedDataset(val_pairs, transform=eval_transform)

        train_loader = DataLoader(
            train_dataset,
            **get_dataloader_kwargs(args, device, shuffle=True)
        )
        val_loader = DataLoader(
            val_dataset,
            **get_dataloader_kwargs(args, device, shuffle=False)
        )

        print(f"Training dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")

        model = create_model(num_classes=args.num_classes).to(device)
        criterion = nn.CrossEntropyLoss()

        if args.weights:
            weights_path = Path(args.weights)
            if not weights_path.exists():
                raise FileNotFoundError(f"Pretrained weights not found: {weights_path}")

            print(f"Loading pretrained weights from: {weights_path}")
            weights_obj = torch.load(weights_path, map_location=device)

            if isinstance(weights_obj, dict) and "model" in weights_obj:
                weights_dict = weights_obj["model"]
            else:
                weights_dict = weights_obj

            # Remove classifier head weights
            for k in list(weights_dict.keys()):
                if "head" in k:
                    del weights_dict[k]

            missing, unexpected = model.load_state_dict(weights_dict, strict=False)
            print(f"Missing keys: {len(missing)}")
            print(f"Unexpected keys: {len(unexpected)}")

        if args.freeze_layers:
            print("Freezing all layers except classifier head.")
            for name, param in model.named_parameters():
                if "head" not in name:
                    param.requires_grad_(False)

        optimizer = optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=args.lr_patience,
        )

        best_model_path = save_dir / f"best_model_fold_{fold}.pth"

        best_val_loss = float("inf")
        best_val_acc_single = 0.0
        best_val_acc_double = 0.0
        best_train_acc_for_best_val = 0.0
        patience_counter = 0

        for epoch in range(args.epochs):
            model.train()

            train_loss, train_acc = train_one_epoch(
                model=model,
                optimizer=optimizer,
                data_loader=train_loader,
                device=device,
                epoch=epoch,
            )

            val_loss, val_acc_single, val_acc_double = evaluate_pairs(
                model=model,
                data_loader=val_loader,
                device=device,
                epoch=epoch,
            )

            writer.add_scalar(f"Fold_{fold}/train_loss", train_loss, epoch)
            writer.add_scalar(f"Fold_{fold}/train_acc", train_acc, epoch)
            writer.add_scalar(f"Fold_{fold}/val_loss", val_loss, epoch)
            writer.add_scalar(f"Fold_{fold}/val_acc_single", val_acc_single, epoch)
            writer.add_scalar(f"Fold_{fold}/val_acc_double", val_acc_double, epoch)
            writer.add_scalar(f"Fold_{fold}/lr", optimizer.param_groups[0]["lr"], epoch)

            scheduler.step(val_loss)

            improved = False

            if val_acc_single > best_val_acc_single:
                improved = True
            elif val_acc_single == best_val_acc_single and train_acc > best_train_acc_for_best_val:
                improved = True

            if improved:
                best_val_loss = val_loss
                best_val_acc_single = val_acc_single
                best_val_acc_double = val_acc_double
                best_train_acc_for_best_val = train_acc
                patience_counter = 0

                torch.save(model.state_dict(), best_model_path)
                print(
                    f"[Fold {fold}] Saved best model at epoch {epoch + 1} | "
                    f"val_acc_single={best_val_acc_single:.4f}, "
                    f"val_acc_double={best_val_acc_double:.4f}"
                )
            else:
                patience_counter += 1
                if patience_counter >= args.early_stop_patience:
                    print(f"[Fold {fold}] Early stopping at epoch {epoch + 1}")
                    break

        # Load best model before testing
        if not best_model_path.exists():
            raise FileNotFoundError(f"Best model was not saved for fold {fold}: {best_model_path}")

        model.load_state_dict(torch.load(best_model_path, map_location=device))

        test_loss, test_acc_single, test_acc_double = evaluate_pairs(
            model=model,
            data_loader=test_loader,
            device=device,
            epoch=0,
        )

        fold_result = {
            "fold": fold,
            "best_val_loss": float(best_val_loss),
            "best_val_acc_single": float(best_val_acc_single),
            "best_val_acc_double": float(best_val_acc_double),
            "best_train_acc": float(best_train_acc_for_best_val),
            "test_loss": float(test_loss),
            "test_acc_single": float(test_acc_single),
            "test_acc_double": float(test_acc_double),
            "best_model_path": str(best_model_path),
        }
        fold_results.append(fold_result)

        print(f"\nFold {fold} Results:")
        print(
            f"Best Validation Loss: {best_val_loss:.4f}, "
            f"Best Single-side Validation Accuracy: {best_val_acc_single:.4f}, "
            f"Best Double-side Validation Accuracy: {best_val_acc_double:.4f}"
        )
        print(f"Corresponding Training Accuracy: {best_train_acc_for_best_val:.4f}")
        print(
            f"Test Loss: {test_loss:.4f}, "
            f"Test Single-side Accuracy: {test_acc_single:.4f}, "
            f"Test Double-side Accuracy: {test_acc_double:.4f}"
        )

    # Average summary
    avg_results = {
        "avg_best_val_loss": float(np.mean([r["best_val_loss"] for r in fold_results])),
        "avg_best_val_acc_single": float(np.mean([r["best_val_acc_single"] for r in fold_results])),
        "avg_best_val_acc_double": float(np.mean([r["best_val_acc_double"] for r in fold_results])),
        "avg_best_train_acc": float(np.mean([r["best_train_acc"] for r in fold_results])),
        "avg_test_loss": float(np.mean([r["test_loss"] for r in fold_results])),
        "avg_test_acc_single": float(np.mean([r["test_acc_single"] for r in fold_results])),
        "avg_test_acc_double": float(np.mean([r["test_acc_double"] for r in fold_results])),
    }

    print("\n" + "=" * 20 + " Average Results over all folds " + "=" * 20)
    print(f"Average Best Validation Loss: {avg_results['avg_best_val_loss']:.4f}")
    print(f"Average Best Single-side Validation Accuracy: {avg_results['avg_best_val_acc_single']:.4f}")
    print(f"Average Best Double-side Validation Accuracy: {avg_results['avg_best_val_acc_double']:.4f}")
    print(f"Average Corresponding Training Accuracy: {avg_results['avg_best_train_acc']:.4f}")
    print(f"Average Test Loss: {avg_results['avg_test_loss']:.4f}")
    print(f"Average Test Single-side Accuracy: {avg_results['avg_test_acc_single']:.4f}")
    print(f"Average Test Double-side Accuracy: {avg_results['avg_test_acc_double']:.4f}")

    # Save config and results
    save_json(vars(args), save_dir / "config.json")
    save_json(
        {
            "fold_results": fold_results,
            "average_results": avg_results,
        },
        save_dir / "results_summary.json",
    )

    writer.close()


def build_parser():
    parser = argparse.ArgumentParser(description="Train Swin Transformer with paired-image evaluation.")

    parser.add_argument("--seed", type=int, default=27, help="Random seed.")
    parser.add_argument("--num_classes", type=int, default=2, help="Number of output classes.")
    parser.add_argument("--epochs", type=int, default=120, help="Maximum number of epochs.")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size.")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay for AdamW.")
    parser.add_argument("--lr-patience", type=int, default=3, help="Patience for ReduceLROnPlateau.")
    parser.add_argument("--early-stop-patience", type=int, default=20, help="Early stopping patience.")
    parser.add_argument("--img-size", type=int, default=224, help="Input image size.")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of folds for K-fold cross-validation.")
    parser.add_argument("--test-size", type=float, default=0.1, help="Proportion of independent test set.")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers.")

    parser.add_argument("--data-path", type=str, default="./未划分数据集", help="Path to dataset root.")
    parser.add_argument(
        "--weights",
        type=str,
        default="swin_base_patch4_window7_224_22k.pth",
        help="Path to pretrained weights. Use empty string to disable.",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default="./weights_seed27",
        help="Directory to save best models and summary files.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="./runs",
        help="Directory to save TensorBoard logs.",
    )
    parser.add_argument(
        "--freeze-layers",
        action="store_true",
        help="Freeze all layers except classification head.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to use, e.g. 'cuda:0' or 'cpu'.",
    )

    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    main(args)