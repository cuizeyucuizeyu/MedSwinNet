import os
import json
import pickle
import random
from typing import Dict, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

import torch
import torch.nn as nn


def read_split_data(root: str, val_rate: float = 0.2, seed: int = 0):
    """
    Split image dataset into training and validation subsets.

    The directory structure is expected to follow ImageFolder convention:
        root/
        ├── class_a/
        ├── class_b/
        └── ...

    Args:
        root: Dataset root directory.
        val_rate: Validation split ratio.
        seed: Random seed for reproducibility.

    Returns:
        train_images_path, train_images_label, val_images_path, val_images_label
    """
    random.seed(seed)
    assert os.path.exists(root), f"Dataset root does not exist: {root}"

    class_names = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    class_names.sort()

    class_indices = {class_name: idx for idx, class_name in enumerate(class_names)}

    json_str = json.dumps({idx: class_name for class_name, idx in class_indices.items()}, indent=4)
    with open("class_indices.json", "w", encoding="utf-8") as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    val_images_path = []
    val_images_label = []
    every_class_num = []

    supported = [".jpg", ".JPG", ".png", ".PNG", ".jpeg", ".JPEG"]

    for cla in class_names:
        cla_path = os.path.join(root, cla)
        images = [
            os.path.join(cla_path, img_name)
            for img_name in os.listdir(cla_path)
            if os.path.splitext(img_name)[-1] in supported
        ]
        images.sort()

        image_class = class_indices[cla]
        every_class_num.append(len(images))

        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print(f"{sum(every_class_num)} images were found in the dataset.")
    print(f"{len(train_images_path)} images for training.")
    print(f"{len(val_images_path)} images for validation.")

    assert len(train_images_path) > 0, "Number of training images must be greater than 0."
    assert len(val_images_path) > 0, "Number of validation images must be greater than 0."

    return train_images_path, train_images_label, val_images_path, val_images_label


def plot_data_loader_image(data_loader):
    """
    Visualize a few images from a DataLoader.
    """
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = "./class_indices.json"
    assert os.path.exists(json_path), f"{json_path} does not exist."

    with open(json_path, "r", encoding="utf-8") as json_file:
        class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data

        for i in range(plot_num):
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255.0
            img = np.clip(img, 0, 255).astype("uint8")

            label = labels[i].item()

            plt.subplot(1, plot_num, i + 1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])
            plt.yticks([])
            plt.imshow(img)

        plt.show()
        break


def write_pickle(list_info: list, file_name: str):
    """
    Save a Python list to a pickle file.
    """
    with open(file_name, "wb") as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    """
    Load a Python list from a pickle file.
    """
    with open(file_name, "rb") as f:
        info_list = pickle.load(f)
    return info_list


def compute_binary_metrics(y_true: Union[List[int], np.ndarray],
                           y_pred: Union[List[int], np.ndarray]) -> Dict[str, float]:
    """
    Compute standard binary classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Dictionary containing F1, precision, recall, and specificity.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    f1 = f1_score(y_true, y_pred, average="binary", zero_division=0)
    recall = recall_score(y_true, y_pred, average="binary", zero_division=0)
    precision = precision_score(y_true, y_pred, average="binary", zero_division=0)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp + 1e-6)

    return {
        "f1_score": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
    }


def train_one_epoch(model,
                    optimizer,
                    data_loader,
                    device,
                    epoch,
                    return_details: bool = False):
    """
    Train the model for one epoch.

    Args:
        model: PyTorch model.
        optimizer: Optimizer.
        data_loader: Training DataLoader.
        device: Device.
        epoch: Current epoch index.
        return_details: If True, also return metrics, labels, and probabilities.

    Returns:
        By default:
            avg_loss, acc

        If return_details=True:
            avg_loss, acc, metrics_dict, all_labels, all_probs
    """
    model.train()
    loss_function = nn.CrossEntropyLoss()

    accu_loss = 0.0
    accu_num = 0
    sample_num = 0

    all_preds = []
    all_labels = []
    all_probs = []

    optimizer.zero_grad()

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} Training")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        sample_num += images.size(0)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        _, preds = torch.max(outputs, dim=1)

        accu_loss += loss.item()
        accu_num += torch.eq(preds, labels).sum().item()

        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        all_probs.append(probs)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = accu_loss / len(data_loader)
    acc = accu_num / sample_num

    if not return_details:
        return avg_loss, acc

    metrics = compute_binary_metrics(all_labels, all_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.asarray(all_labels)

    return avg_loss, acc, metrics, all_labels, all_probs


@torch.no_grad()
def evaluate(model,
             data_loader,
             device,
             epoch,
             return_details: bool = False):
    """
    Evaluate a standard single-image classification model.

    Args:
        model: PyTorch model.
        data_loader: Validation DataLoader.
        device: Device.
        epoch: Current epoch index.
        return_details: If True, also return metrics, labels, and probabilities.

    Returns:
        By default:
            avg_loss, acc

        If return_details=True:
            avg_loss, acc, metrics_dict, all_labels, all_probs
    """
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    accu_loss = 0.0
    accu_num = 0
    sample_num = 0

    all_preds = []
    all_labels = []
    all_probs = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} Validation")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        sample_num += images.size(0)

        outputs = model(images)
        loss = loss_function(outputs, labels)

        _, preds = torch.max(outputs, dim=1)

        accu_loss += loss.item()
        accu_num += torch.eq(preds, labels).sum().item()

        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        all_probs.append(probs)

        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())

    avg_loss = accu_loss / len(data_loader)
    acc = accu_num / sample_num

    if not return_details:
        return avg_loss, acc

    metrics = compute_binary_metrics(all_labels, all_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.asarray(all_labels)

    return avg_loss, acc, metrics, all_labels, all_probs


@torch.no_grad()
def evaluate_pairs(model,
                   data_loader,
                   device,
                   epoch,
                   return_details: bool = False,
                   double_strategy: str = "mean_pred_floor"):
    """
    Evaluate paired left/right images.

    Each sample contains:
        left_image, right_image, label, severe_side

    Args:
        model: PyTorch model.
        data_loader: Paired validation/test DataLoader.
        device: Device.
        epoch: Current epoch index.
        return_details: If True, also return metrics, labels, and probabilities.
        double_strategy:
            - "mean_pred_floor": reproduce the original logic:
                preds_double = (preds_left + preds_right) // 2
            - "mean_logits": average logits from left/right, then argmax

    Returns:
        By default:
            avg_loss, acc_single, acc_double

        If return_details=True:
            avg_loss, acc_single, acc_double,
            metrics_single, metrics_double,
            all_labels_single, all_probs_single
    """
    model.eval()
    loss_function = nn.CrossEntropyLoss()

    accu_loss = 0.0
    accu_num_single = 0
    accu_num_double = 0
    sample_num = 0

    all_preds_single = []
    all_labels_single = []
    all_probs_single = []

    all_preds_double = []
    all_labels_double = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} Validation")

    for left_images, right_images, labels, severe_sides in progress_bar:
        left_images = left_images.to(device)
        right_images = right_images.to(device)
        labels = labels.to(device)

        sample_num += labels.size(0)

        outputs_left = model(left_images)
        outputs_right = model(right_images)

        _, preds_left = torch.max(outputs_left, dim=1)
        _, preds_right = torch.max(outputs_right, dim=1)

        # Single-side prediction is based on the left image to preserve original logic
        probs_left = torch.softmax(outputs_left, dim=1).detach().cpu().numpy()
        all_probs_single.append(probs_left)

        all_preds_single.extend(preds_left.detach().cpu().numpy())
        all_labels_single.extend(labels.detach().cpu().numpy())

        # Paired prediction
        if double_strategy == "mean_pred_floor":
            preds_double = (preds_left + preds_right) // 2
        elif double_strategy == "mean_logits":
            outputs_mean = (outputs_left + outputs_right) / 2.0
            preds_double = torch.argmax(outputs_mean, dim=1)
        else:
            raise ValueError(f"Unsupported double_strategy: {double_strategy}")

        all_preds_double.extend(preds_double.detach().cpu().numpy())
        all_labels_double.extend(labels.detach().cpu().numpy())

        accu_num_single += torch.eq(preds_left, labels).sum().item()
        accu_num_double += torch.eq(preds_double, labels).sum().item()

        loss_left = loss_function(outputs_left, labels)
        loss_right = loss_function(outputs_right, labels)
        loss = (loss_left + loss_right) / 2.0
        accu_loss += loss.item()

    avg_loss = accu_loss / len(data_loader)
    acc_single = accu_num_single / sample_num
    acc_double = accu_num_double / sample_num

    if not return_details:
        return avg_loss, acc_single, acc_double

    metrics_single = compute_binary_metrics(all_labels_single, all_preds_single)
    metrics_double = compute_binary_metrics(all_labels_double, all_preds_double)

    all_labels_single = np.asarray(all_labels_single)
    all_probs_single = np.concatenate(all_probs_single, axis=0)

    return (
        avg_loss,
        acc_single,
        acc_double,
        metrics_single,
        metrics_double,
        all_labels_single,
        all_probs_single,
    )


@torch.no_grad()
def evaluate_bce(model,
                 data_loader,
                 device,
                 epoch,
                 return_details: bool = False):
    """
    Evaluate a model trained with BCEWithLogitsLoss for binary classification.

    Args:
        model: PyTorch model.
        data_loader: Validation DataLoader.
        device: Device.
        epoch: Current epoch index.
        return_details: If True, also return metrics, labels, and probabilities.

    Returns:
        By default:
            avg_loss, acc

        If return_details=True:
            avg_loss, acc, metrics_dict, all_labels, all_probs
    """
    model.eval()
    loss_function = nn.BCEWithLogitsLoss()

    accu_loss = 0.0
    accu_num = 0
    sample_num = 0

    all_preds = []
    all_labels = []
    all_probs = []

    progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1} Validation (BCE)")

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        sample_num += images.size(0)

        outputs = model(images)

        labels_one_hot = torch.zeros(labels.size(0), 2, device=device)
        labels_one_hot.scatter_(1, labels.unsqueeze(1), 1)

        loss = loss_function(outputs, labels_one_hot.float())
        accu_loss += loss.item()

        probs = torch.sigmoid(outputs)
        pred_classes = (probs > 0.5).int().argmax(dim=1)

        accu_num += torch.eq(pred_classes, labels).sum().item()

        all_preds.extend(pred_classes.detach().cpu().numpy())
        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.append(probs.detach().cpu().numpy())

    avg_loss = accu_loss / len(data_loader)
    acc = accu_num / sample_num

    if not return_details:
        return avg_loss, acc

    metrics = compute_binary_metrics(all_labels, all_preds)
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.asarray(all_labels)

    return avg_loss, acc, metrics, all_labels, all_probs