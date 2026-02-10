import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ConvBlock(nn.Module):
    """Simple conv block with BN + ReLU, optional dropout."""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))  # dropout active only in train mode
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetResNet50(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, pretrained=True, dropout=0.5):
        super().__init__()
        # --- Encoder (same as before) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        self.input_layer = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        # --- Decoder with dropout ---
        self.up4 = nn.ConvTranspose2d(2048, 1024, kernel_size=2, stride=2)
        self.dec4 = ConvBlock(2048, 1024, dropout=dropout)

        self.up3 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(1024, 512, dropout=dropout)

        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(512, 256, dropout=dropout)

        self.up1 = nn.ConvTranspose2d(256, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(128, 64, dropout=dropout)

        self.up0 = nn.ConvTranspose2d(64, 64, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x0 = self.input_layer(x)
        x1 = self.maxpool(x0)
        x1 = self.encoder1(x1)
        x2 = self.encoder2(x1)
        x3 = self.encoder3(x2)
        x4 = self.encoder4(x3)

        # Decoder
        d4 = self.up4(x4)
        d4 = self.dec4(torch.cat([d4, x3], dim=1))

        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, x2], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, x1], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, x0], dim=1))

        out = self.up0(d1)
        out = self.final_conv(out)
        return out

class UnifiedDiceCELoss(nn.Module):
    """
    Combined Dice + Cross-Entropy loss for binary or multi-class segmentation.
    Automatically detects number of classes from preds.shape[1].
    """
    def __init__(self, dice_weight=0.7, ce_weight=0.3, smooth=1e-6):
        super().__init__()
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight
        self.smooth = smooth
        self.ce_loss_binary = nn.BCEWithLogitsLoss()
        self.ce_loss_multiclass = nn.CrossEntropyLoss()

    def forward(self, preds, targets):
        """
        preds: raw logits, shape [B, C, H, W]
        targets: 
            - binary case: [B, 1, H, W] (values 0/1)
            - multi-class case: [B, H, W] with class indices
        """
        num_classes = preds.shape[1]

        if num_classes == 1:
            # Binary segmentation
            bce = self.ce_loss_binary(preds, targets.float())  # [B,1,H,W] vs [B,1,H,W]
            dice = self.binary_dice_loss(preds, targets)
            loss = self.dice_weight * dice + self.ce_weight * bce
        else:
            # Multi-class segmentation
            ce = self.ce_loss_multiclass(preds, targets)  # preds [B,C,H,W], targets [B,H,W]
            dice = self.multiclass_dice_loss(preds, targets)
            loss = self.dice_weight * dice + self.ce_weight * ce

        return loss

    def binary_dice_loss(self, preds, targets):
        preds = torch.sigmoid(preds)
        preds_flat = preds.view(-1)
        targets_flat = targets.view(-1).float()
        intersection = (preds_flat * targets_flat).sum()
        union = preds_flat.sum() + targets_flat.sum()
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score

    def multiclass_dice_loss(self, preds, targets):
        preds = F.softmax(preds, dim=1)
        num_classes = preds.shape[1]
        # One-hot encode targets
        targets_onehot = F.one_hot(targets, num_classes=num_classes)  # [B,H,W,C]
        targets_onehot = targets_onehot.permute(0, 3, 1, 2).float()   # [B,C,H,W]
        # Flatten
        preds_flat = preds.contiguous().view(preds.shape[0], num_classes, -1)
        targets_flat = targets_onehot.contiguous().view(targets.shape[0], num_classes, -1)
        intersection = (preds_flat * targets_flat).sum(-1)
        union = preds_flat.sum(-1) + targets_flat.sum(-1)
        dice_score = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice_score.mean()