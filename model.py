import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from config import ENCODER_NAME


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred         = torch.sigmoid(pred).view(-1)
        target       = target.view(-1)
        intersection = (pred * target).sum()
        return 1 - (2. * intersection + 1e-5) / (pred.sum() + target.sum() + 1e-5)


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = DiceLoss()
        self.bce  = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return 0.5 * self.dice(pred, target) + 0.5 * self.bce(pred, target)


def dice_coefficient(pred, target):
    pred         = (torch.sigmoid(pred) > 0.5).float().view(-1)
    target       = target.view(-1)
    intersection = (pred * target).sum()
    return ((2. * intersection + 1e-5) / (pred.sum() + target.sum() + 1e-5)).item()


def iou_score(pred, target):
    pred         = (torch.sigmoid(pred) > 0.5).float().view(-1)
    target       = target.view(-1)
    intersection = (pred * target).sum()
    union        = pred.sum() + target.sum() - intersection
    return ((intersection + 1e-5) / (union + 1e-5)).item()


def build_model(device):
    model = smp.Unet(
        encoder_name=ENCODER_NAME,
        encoder_weights='imagenet',
        in_channels=1,
        classes=1,
        activation=None,
    ).to(device)
    return model
