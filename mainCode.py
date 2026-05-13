"""
BUS-BRA Training with MobileNetV3-Large Encoder
500 Epochs, Batch Size 8, Split 70/20/10
"""

import os
import time
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import albumentations as A
import segmentation_models_pytorch as smp
from tqdm import tqdm
import matplotlib.pyplot as plt


BASE_DIR = r"D:\Capstone project\BUSBRA"
IMG_DIR = os.path.join(BASE_DIR, "Images")
MASK_DIR = os.path.join(BASE_DIR, "Masks")
CSV_PATH = os.path.join(BASE_DIR, "bus_data.csv")

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")

ENCODER_NAME = 'timm-mobilenetv3_large_100'
IMAGE_SIZE = (256, 256)
BATCH_SIZE = 8           # ← CHANGED from 16 to 8
NUM_EPOCHS = 500
LEARNING_RATE = 1e-4

TRAIN_RATIO = 0.70       # ← 70%
VAL_RATIO = 0.20         # ← CHANGED from 0.15 to 0.20 (20%)
TEST_RATIO = 0.10        # ← CHANGED from 0.15 to 0.10 (10%)
RANDOM_SEED = 42


class BUSBRADataset(Dataset):
    def __init__(self, images_path, masks_path, size=(256, 256), transform=None):
        self.images_path = images_path
        self.masks_path = masks_path
        self.size = size
        self.transform = transform
    
    def __len__(self):
        return len(self.images_path)
    
    def __getitem__(self, index):
        image = cv2.imread(self.images_path[index], cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(self.masks_path[index], cv2.IMREAD_GRAYSCALE)
        
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]
        
        image = cv2.resize(image, self.size)
        mask = cv2.resize(mask, self.size, interpolation=cv2.INTER_NEAREST)
        
        image = image.astype(np.float32) / 255.0
        mask = (mask > 0).astype(np.float32)
        
        image = np.expand_dims(image, axis=0)
        mask = np.expand_dims(mask, axis=0)
        
        return torch.from_numpy(image), torch.from_numpy(mask)


def load_data(csv_path, img_dir, mask_dir):
    df = pd.read_csv(csv_path)
    unique_cases = df["Case"].unique()
    
    train_cases, temp_cases = train_test_split(
        unique_cases, test_size=(VAL_RATIO + TEST_RATIO), random_state=RANDOM_SEED
    )
    val_test_ratio = TEST_RATIO / (VAL_RATIO + TEST_RATIO)
    val_cases, test_cases = train_test_split(
        temp_cases, test_size=val_test_ratio, random_state=RANDOM_SEED
    )
    
    def get_paths(subset_df):
        img_paths, mask_paths = [], []
        for _, row in subset_df.iterrows():
            img_id = row["ID"]
            img_path = os.path.join(img_dir, img_id + ".png")
            mask_path = os.path.join(mask_dir, img_id.replace("bus_", "mask_") + ".png")
            if os.path.exists(img_path) and os.path.exists(mask_path):
                img_paths.append(img_path)
                mask_paths.append(mask_path)
        return img_paths, mask_paths
    
    train_x, train_y = get_paths(df[df["Case"].isin(train_cases)])
    val_x, val_y = get_paths(df[df["Case"].isin(val_cases)])
    test_x, test_y = get_paths(df[df["Case"].isin(test_cases)])
    
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred = pred.view(-1)
        target = target.view(-1)
        intersection = (pred * target).sum()
        dice = (2. * intersection + 1e-5) / (pred.sum() + target.sum() + 1e-5)
        return 1 - dice


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
    
    def forward(self, pred, target):
        return 0.5 * self.dice(pred, target) + 0.5 * self.bce(pred, target)


def dice_coefficient(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + 1e-5) / (pred.sum() + target.sum() + 1e-5)
    return dice.item()


def iou_score(pred, target):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-5) / (union + 1e-5)
    return iou.item()


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    epoch_loss = 0.0
    epoch_dice = 0.0
    
    pbar = tqdm(loader, desc='Training')
    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()
        
        dice = dice_coefficient(outputs, masks)
        epoch_loss += loss.item()
        epoch_dice += dice
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    return epoch_loss / len(loader), epoch_dice / len(loader)


def validate_epoch(model, loader, loss_fn, device):
    model.eval()
    epoch_loss = 0.0
    epoch_dice = 0.0
    epoch_iou = 0.0
    
    pbar = tqdm(loader, desc='Validation')
    with torch.no_grad():
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            
            dice = dice_coefficient(outputs, masks)
            iou = iou_score(outputs, masks)
            
            epoch_loss += loss.item()
            epoch_dice += dice
            epoch_iou += iou
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'dice': f'{dice:.4f}'})
    
    return epoch_loss / len(loader), epoch_dice / len(loader), epoch_iou / len(loader)


def plot_history(history, save_dir, show=False):
    epochs     = [h['epoch']      for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss   = [h['val_loss']   for h in history]
    train_dice = [h['train_dice'] for h in history]
    val_dice   = [h['val_dice']   for h in history]
    val_iou    = [h['val_iou']    for h in history]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'MobileNetV3-Large Training (Epoch {epochs[-1]})', fontsize=13, fontweight='bold')

    axes[0].plot(epochs, train_loss, label='Train Loss', color='blue')
    axes[0].plot(epochs, val_loss,   label='Val Loss',   color='orange')
    axes[0].set_title('Loss vs Epochs')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_dice, label='Train Dice', color='green')
    axes[1].plot(epochs, val_dice,   label='Val Dice',   color='red')
    axes[1].set_title('Dice vs Epochs')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(epochs, val_iou, label='Val IoU', color='purple')
    axes[2].set_title('IoU vs Epochs')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU Score')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'training_graph_epoch_{epochs[-1]:03d}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Graph saved: {save_path}")
    if show:
        plt.show()
    plt.close()


def main():
    print("=" * 60)
    print("BUS-BRA TRAINING - MobileNetV3-Large")
    print(f"Epochs: {NUM_EPOCHS} | Batch Size: {BATCH_SIZE}")
    print(f"Split: Train {int(TRAIN_RATIO*100)}% | Val {int(VAL_RATIO*100)}% | Test {int(TEST_RATIO*100)}%")
    print("=" * 60)
    
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(RANDOM_SEED)
    
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\nLoading data...")
    (train_x, train_y), (val_x, val_y), (test_x, test_y) = load_data(CSV_PATH, IMG_DIR, MASK_DIR)
    print(f"Train: {len(train_x)} | Val: {len(val_x)} | Test: {len(test_x)}")
    
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.RandomBrightnessContrast(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.0, rotate_limit=0, p=0.5, border_mode=cv2.BORDER_CONSTANT),
        A.Affine(shear=(-15, 15), p=0.3, mode=cv2.BORDER_CONSTANT),
        A.RandomScale(scale_limit=0.2, p=0.3),
    ])
    
    train_dataset = BUSBRADataset(train_x, train_y, IMAGE_SIZE, train_transform)
    val_dataset = BUSBRADataset(val_x, val_y, IMAGE_SIZE, None)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    
    print("\nBuilding model...")
    print(f"Encoder: {ENCODER_NAME}")
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights='imagenet',
        in_channels=1,
        classes=1,
        activation=None
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    loss_fn = DiceBCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    best_val_dice = 0.0
    best_epoch = 0
    
    history = []
    
    print("\nStarting training...")
    print(f"Total epochs: {NUM_EPOCHS}")
    print("=" * 60)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        start_time = time.time()
        
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice, val_iou = validate_epoch(model, val_loader, loss_fn, device)
        
        elapsed = time.time() - start_time
        mins = int(elapsed / 60)
        secs = int(elapsed - (mins * 60))
        
        print(f"\nEpoch [{epoch:03d}/{NUM_EPOCHS}] {mins}m {secs}s")
        print(f"  Train Dice: {train_dice:.4f}")
        print(f"  Val Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")
        
        scheduler.step(val_loss)
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'best_model.pth'))
            print(f"  ★ New best Val Dice: {val_dice:.4f}")
        
        history.append({
            'epoch':      epoch,
            'train_loss': train_loss,
            'val_loss':   val_loss,
            'train_dice': train_dice,
            'val_dice':   val_dice,
            'val_iou':    val_iou
        })

        if epoch % 5 == 0:
            plot_history(history, CHECKPOINT_DIR)

        if epoch % 10 == 0:
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f'model_epoch_{epoch}.pth'))
    
    torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, 'final_model.pth'))

    plot_history(history, CHECKPOINT_DIR, show=True)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest Val Dice: {best_val_dice:.4f} at epoch {best_epoch}")
    
    import json
    with open(os.path.join(CHECKPOINT_DIR, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nTraining history saved to: {CHECKPOINT_DIR}/training_history.json")


if __name__ == "__main__":
    main()