# ユーティリティ関数の定義
import sys, os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

def tensor_to_image(x):
    x = x * torch.Tensor([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    x = x + torch.Tensor([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
    return transforms.ToPILImage()(x)

def create_confusion_matrix(cm, labels):
    sns.set()
    
    df = pd.DataFrame(cm)
    df.index = labels
    df.columns = labels

    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(df, annot=True, fmt="d", linewidths=.5, ax=ax, cmap="YlGnBu")

def plot_confusion_matrix(model, dataset):
    images_so_far = 0
    fig = plt.figure()

    targets = np.array([])
    preds = np.array([])

    model.to("cuda")
    model.eval()
    with torch.no_grad():
        for img, target in DataLoader(dataset, shuffle=False, batch_size=64, num_workers=4):
            img = img.to("cuda")
            target = target.to("cuda")
            pred = model(img)
            pred = pred.argmax(dim=1)

            targets = np.append(targets, target.cpu().data.numpy())
            preds = np.append(preds, pred.cpu().numpy())
    
    cm = confusion_matrix(targets, preds)
    create_confusion_matrix(cm, dataset.classes)

# データ拡張の定義
augment_transform = transforms.Compose([
    transforms.RandomResizedCrop(size=224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
evaluate_transform = transforms.Compose([
    transforms.Resize(size=224),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# TODO : DATASET_DIR の設定
DATASET_DIR = ""

train_dataset = ImageFolder(f"{DATASET_DIR}/Train", transform=augment_transform)
val_dataset = ImageFolder(f"{DATASET_DIR}/Validation", transform=evaluate_transform)
test_dataset = ImageFolder(f"{DATASET_DIR}/Test", transform=evaluate_transform)

for img, target in DataLoader(train_dataset, shuffle = True, batch_size = 1850):
    x_train = img.numpy()
    y_train = target.numpy()
    # x_train = x_train.transpose((1, 2, 3, 0))

for img, target in DataLoader(test_dataset, shuffle = True, batch_size = 3690):
    x_test = img.numpy()
    y_test = target.numpy()
    # x_test = x_test.transpose((1, 2, 3, 0))

"""
# validation データ作成
for img, target in DataLoader(val_dataset, shuffle = True, batch_size = 1846):
    x_val = img.numpy()
    y_val = target.numpy()
    x_val = x_val.transpose((1, 2, 3, 0))
"""