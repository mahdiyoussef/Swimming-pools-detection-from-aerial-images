"""
Custom Layers and Loss Functions for Swimming Pool Detection System.

This module provides optional custom modifications including
attention mechanisms, custom loss functions, and architecture enhancements.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import logging
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.

    Applies spatial attention to feature maps to focus on
    relevant regions for pool detection.

    Attributes:
        kernel_size: Size of convolution kernel for attention.
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """
        Initialize Spatial Attention module.

        Args:
            kernel_size: Size of the convolution kernel (must be odd).
        """
        super().__init__()
        
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        
        self.conv = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to input tensor.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            Attention-weighted feature tensor of same shape.
        """
        # Compute channel-wise statistics
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and convolve
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation style).

    Applies channel-wise attention to emphasize important
    feature channels.

    Attributes:
        reduction: Reduction ratio for the bottleneck.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """
        Initialize Channel Attention module.

        Args:
            channels: Number of input channels.
            reduction: Channel reduction ratio for bottleneck.
        """
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input tensor.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            Attention-weighted feature tensor of same shape.
        """
        batch_size, channels, _, _ = x.size()
        
        # Average pooling branch
        avg_out = self.avg_pool(x).view(batch_size, channels)
        avg_out = self.mlp(avg_out)
        
        # Max pooling branch
        max_out = self.max_pool(x).view(batch_size, channels)
        max_out = self.mlp(max_out)
        
        # Combine and apply attention
        attention = self.sigmoid(avg_out + max_out)
        attention = attention.view(batch_size, channels, 1, 1)
        
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).

    Combines channel and spatial attention for enhanced
    feature representation.

    Attributes:
        channel_attention: Channel attention module.
        spatial_attention: Spatial attention module.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 16,
        kernel_size: int = 7
    ) -> None:
        """
        Initialize CBAM module.

        Args:
            channels: Number of input channels.
            reduction: Reduction ratio for channel attention.
            kernel_size: Kernel size for spatial attention.
        """
        super().__init__()
        
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM attention to input tensor.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            Attention-enhanced feature tensor of same shape.
        """
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.

    Reduces the loss contribution from easy examples and
    focuses training on hard examples.

    Attributes:
        alpha: Weighting factor for positive class.
        gamma: Focusing parameter (gamma >= 0).
        reduction: Reduction method ('none', 'mean', 'sum').
    """

    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        reduction: str = "mean"
    ) -> None:
        """
        Initialize Focal Loss.

        Args:
            alpha: Weighting factor for positive examples.
            gamma: Focusing parameter to down-weight easy examples.
            reduction: How to reduce the loss ('none', 'mean', 'sum').
        """
        super().__init__()
        
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss.

        Args:
            predictions: Predicted logits of shape (N, C) or (N,).
            targets: Ground truth labels of shape (N,).

        Returns:
            Focal loss value.
        """
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) Loss for bounding box regression.

    Computes loss based on IoU between predicted and target boxes.

    Attributes:
        loss_type: Type of IoU loss ('iou', 'giou', 'diou', 'ciou').
    """

    def __init__(self, loss_type: str = "ciou") -> None:
        """
        Initialize IoU Loss.

        Args:
            loss_type: Type of IoU loss to compute.
        """
        super().__init__()
        
        valid_types = ["iou", "giou", "diou", "ciou"]
        if loss_type not in valid_types:
            raise ValueError(f"loss_type must be one of {valid_types}")
        
        self.loss_type = loss_type

    def forward(
        self,
        pred_boxes: torch.Tensor,
        target_boxes: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute IoU-based loss.

        Args:
            pred_boxes: Predicted boxes (N, 4) in x1y1x2y2 format.
            target_boxes: Target boxes (N, 4) in x1y1x2y2 format.

        Returns:
            IoU loss value.
        """
        iou = self._compute_iou(pred_boxes, target_boxes)
        
        if self.loss_type == "iou":
            loss = 1 - iou
        elif self.loss_type == "giou":
            loss = 1 - self._compute_giou(pred_boxes, target_boxes)
        elif self.loss_type == "diou":
            loss = 1 - self._compute_diou(pred_boxes, target_boxes)
        else:  # ciou
            loss = 1 - self._compute_ciou(pred_boxes, target_boxes)
        
        return loss.mean()

    def _compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Compute standard IoU between box sets."""
        # Intersection
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Union
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union_area = area1 + area2 - inter_area
        
        return inter_area / (union_area + 1e-7)

    def _compute_giou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Compute Generalized IoU."""
        iou = self._compute_iou(boxes1, boxes2)
        
        # Enclosing box
        enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        
        # Union area
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = area1 + area2 - inter_area
        
        giou = iou - (enc_area - union_area) / (enc_area + 1e-7)
        
        return giou

    def _compute_diou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Compute Distance IoU."""
        iou = self._compute_iou(boxes1, boxes2)
        
        # Center distance
        center1_x = (boxes1[:, 0] + boxes1[:, 2]) / 2
        center1_y = (boxes1[:, 1] + boxes1[:, 3]) / 2
        center2_x = (boxes2[:, 0] + boxes2[:, 2]) / 2
        center2_y = (boxes2[:, 1] + boxes2[:, 3]) / 2
        
        center_dist = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
        
        # Enclosing box diagonal
        enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
        
        diou = iou - center_dist / (enc_diag + 1e-7)
        
        return diou

    def _compute_ciou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """Compute Complete IoU."""
        import math
        
        diou = self._compute_diou(boxes1, boxes2)
        
        # Aspect ratio consistency
        w1 = boxes1[:, 2] - boxes1[:, 0]
        h1 = boxes1[:, 3] - boxes1[:, 1]
        w2 = boxes2[:, 2] - boxes2[:, 0]
        h2 = boxes2[:, 3] - boxes2[:, 1]
        
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(w2 / (h2 + 1e-7)) - torch.atan(w1 / (h1 + 1e-7)), 2
        )
        
        iou = self._compute_iou(boxes1, boxes2)
        alpha = v / (1 - iou + v + 1e-7)
        
        ciou = diou - alpha * v
        
        return ciou


class PoolShapeClassifier(nn.Module):
    """
    Auxiliary classifier for pool shape prediction.

    Classifies detected pools into shape categories
    (rectangular, oval, irregular).

    Attributes:
        num_classes: Number of shape classes.
    """

    def __init__(
        self,
        input_features: int = 256,
        num_classes: int = 3
    ) -> None:
        """
        Initialize Pool Shape Classifier.

        Args:
            input_features: Number of input features from backbone.
            num_classes: Number of shape categories.
        """
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(input_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
        
        self.class_names = ["rectangular", "oval", "irregular"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify pool shape from features.

        Args:
            x: Input feature tensor of shape (B, C, H, W).

        Returns:
            Shape logits of shape (B, num_classes).
        """
        return self.classifier(x)


def main() -> None:
    """Test custom layers."""
    logger.info("Testing custom layers...")
    
    # Test CBAM
    cbam = CBAM(channels=64)
    x = torch.randn(2, 64, 32, 32)
    out = cbam(x)
    logger.info(f"CBAM: input {x.shape} -> output {out.shape}")
    
    # Test Focal Loss
    focal = FocalLoss()
    preds = torch.randn(10, 2)
    targets = torch.randint(0, 2, (10,))
    loss = focal(preds, targets)
    logger.info(f"Focal Loss: {loss.item():.4f}")
    
    # Test IoU Loss
    iou_loss = IoULoss(loss_type="ciou")
    boxes1 = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32)
    boxes2 = torch.tensor([[15, 15, 55, 55], [25, 25, 65, 65]], dtype=torch.float32)
    loss = iou_loss(boxes1, boxes2)
    logger.info(f"CIoU Loss: {loss.item():.4f}")
    
    logger.info("All tests passed.")


if __name__ == "__main__":
    main()
