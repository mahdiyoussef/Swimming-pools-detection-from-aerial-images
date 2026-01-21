"""
Custom Layers and Loss Functions for Swimming Pool Detection System.
=====================================================================

This module provides custom neural network components including attention
mechanisms, specialized loss functions, and auxiliary classifiers for
enhanced swimming pool detection in aerial imagery.

Architecture Overview:
    ┌─────────────────────────────────────────────────────────────────┐
    │                    YOLO26 + Custom Layers                       │
    ├─────────────────────────────────────────────────────────────────┤
    │                                                                  │
    │   Input Image (512x512x3)                                       │
    │         │                                                        │
    │         ▼                                                        │
    │   ┌─────────────┐                                               │
    │   │  Backbone   │  Standard YOLO26 CSPDarknet                  │
    │   └─────────────┘                                               │
    │         │                                                        │
    │         ▼                                                        │
    │   ┌─────────────┐                                               │
    │   │    CBAM     │  ◄── Custom Attention Module                  │
    │   │  Attention  │      (Channel + Spatial Attention)            │
    │   └─────────────┘                                               │
    │         │                                                        │
    │         ▼                                                        │
    │   ┌─────────────┐                                               │
    │   │    Head     │  Detection outputs                            │
    │   └─────────────┘                                               │
    │         │                                                        │
    │         ▼                                                        │
    │   ┌─────────────┐                                               │
    │   │ CIoU Loss   │  ◄── Custom Loss Function                     │
    │   │ Focal Loss  │      (Better localization & class imbalance)  │
    │   └─────────────┘                                               │
    │                                                                  │
    └─────────────────────────────────────────────────────────────────┘

Custom Components:
    1. CBAM (Convolutional Block Attention Module):
       - ChannelAttention: "What" to focus on (feature channels)
       - SpatialAttention: "Where" to focus on (spatial locations)
    
    2. Loss Functions:
       - FocalLoss: Handles class imbalance (pools are sparse in images)
       - IoULoss: Better bounding box regression (IoU, GIoU, DIoU, CIoU)
    
    3. PoolShapeClassifier: Auxiliary head for pool shape classification

Usage:
    # Add attention to feature maps
    attention = CBAM(channels=256, reduction=16)
    enhanced_features = attention(features)
    
    # Custom loss
    focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    loss = focal_loss(predictions, targets)

References:
    - CBAM: "CBAM: Convolutional Block Attention Module" (Woo et al., 2018)
    - Focal Loss: "Focal Loss for Dense Object Detection" (Lin et al., 2017)
    - CIoU: "Distance-IoU Loss" (Zheng et al., 2020)

Author: Swimming Pool Detection Team
Date: 2026-01-02
Version: 1.0.0
"""

# =============================================================================
# IMPORTS
# =============================================================================

import logging                    # Logging functionality
from typing import Optional, Tuple  # Type hints

import torch                      # PyTorch tensor library
import torch.nn as nn            # Neural network modules
import torch.nn.functional as F  # Functional operations

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# =============================================================================
# ATTENTION MECHANISMS
# =============================================================================
# Attention mechanisms help the model focus on relevant features.
# They learn to emphasize important channels and spatial locations
# while suppressing less informative ones.
# =============================================================================

class SpatialAttention(nn.Module):
    """
    Spatial Attention Module.
    
    This module learns WHERE in the image to focus attention.
    It computes attention weights for each spatial location (H x W)
    by analyzing channel statistics at each position.
    
    How it works:
        1. Compute channel-wise average and max at each spatial location
        2. Concatenate these two spatial maps
        3. Apply convolution to learn attention weights
        4. Multiply with input features
    
    Visual Representation:
        Input Feature Map           Attention Map         Output
        (C x H x W)                 (1 x H x W)          (C x H x W)
        ┌─────────┐                 ┌─────────┐          ┌─────────┐
        │ ███████ │                 │ ░░▓▓▓░░ │          │ ░░███░░ │
        │ ███████ │  ──► Compute ──►│ ░▓▓▓▓▓░ │ ──► × ──►│ ░█████░ │
        │ ███████ │      Attention  │ ░░▓▓▓░░ │          │ ░░███░░ │
        └─────────┘                 └─────────┘          └─────────┘
        
        Where darker = higher attention weight
    
    Attributes:
        conv (nn.Conv2d): Convolution layer for attention computation.
        sigmoid (nn.Sigmoid): Activation to get weights in [0, 1].
    
    Example:
        >>> spatial_attn = SpatialAttention(kernel_size=7)
        >>> features = torch.randn(2, 64, 32, 32)  # B, C, H, W
        >>> output = spatial_attn(features)  # Same shape
    """

    def __init__(self, kernel_size: int = 7) -> None:
        """
        Initialize Spatial Attention module.
        
        Args:
            kernel_size (int): Size of the convolution kernel.
                Must be an odd number for symmetric padding.
                Default: 7 (captures 7x7 spatial context).
        
        Raises:
            ValueError: If kernel_size is even.
        """
        super().__init__()
        
        # Validate kernel size (must be odd for symmetric padding)
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        
        # Convolution layer: 2 channels (avg + max) -> 1 channel (attention)
        # Uses same padding to preserve spatial dimensions
        self.conv = nn.Conv2d(
            in_channels=2,                    # Average + Max pooled channels
            out_channels=1,                   # Single attention map
            kernel_size=kernel_size,          # Spatial context size
            padding=kernel_size // 2,         # Same padding
            bias=False                        # No bias needed
        )
        
        # Sigmoid to ensure attention weights are in [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply spatial attention to input feature tensor.
        
        Computation:
            1. avg_out = mean(x, dim=channel)  -> (B, 1, H, W)
            2. max_out = max(x, dim=channel)   -> (B, 1, H, W)
            3. combined = concat([avg, max])   -> (B, 2, H, W)
            4. attention = sigmoid(conv(combined)) -> (B, 1, H, W)
            5. output = x * attention          -> (B, C, H, W)
        
        Args:
            x (torch.Tensor): Input feature tensor of shape (B, C, H, W).
                - B: Batch size
                - C: Number of channels
                - H: Height
                - W: Width
        
        Returns:
            torch.Tensor: Attention-weighted features of same shape (B, C, H, W).
                Features at important spatial locations are emphasized.
        """
        # =====================================================================
        # Step 1: Compute channel-wise statistics at each spatial location
        # =====================================================================
        
        # Average pooling across channels: captures overall feature response
        # Shape: (B, C, H, W) -> (B, 1, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        
        # Max pooling across channels: captures strongest feature response
        # Shape: (B, C, H, W) -> (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # =====================================================================
        # Step 2: Combine statistics and compute attention
        # =====================================================================
        
        # Concatenate along channel dimension
        # Shape: (B, 2, H, W)
        combined = torch.cat([avg_out, max_out], dim=1)
        
        # Apply convolution and sigmoid to get attention weights
        # Shape: (B, 2, H, W) -> (B, 1, H, W)
        attention = self.sigmoid(self.conv(combined))
        
        # =====================================================================
        # Step 3: Apply attention to input features
        # =====================================================================
        
        # Element-wise multiplication (broadcasting across channels)
        # Shape: (B, C, H, W) * (B, 1, H, W) -> (B, C, H, W)
        return x * attention


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation style).
    
    This module learns WHAT (which feature channels) to focus on.
    It computes attention weights for each channel by analyzing
    global spatial information.
    
    How it works:
        1. Squeeze: Global average and max pooling (H,W -> 1,1)
        2. Excitation: Shared MLP to learn channel importance
        3. Apply: Multiply attention weights with channels
    
    Mathematical Formulation:
        Mc(F) = σ(MLP(AvgPool(F)) + MLP(MaxPool(F)))
        
        Where:
        - F: Input feature map (C x H x W)
        - σ: Sigmoid activation
        - MLP: Two fully-connected layers with reduction
    
    Architecture Diagram:
        Input Features (C x H x W)
               │
               ├──────────────┬──────────────┐
               ▼              ▼              │
        ┌─────────────┐ ┌─────────────┐     │
        │ AvgPool     │ │ MaxPool     │     │
        │ (C x 1 x 1) │ │ (C x 1 x 1) │     │
        └─────────────┘ └─────────────┘     │
               │              │              │
               ▼              ▼              │
        ┌─────────────────────────────┐     │
        │       Shared MLP            │     │
        │  FC(C -> C/r) -> ReLU       │     │
        │  FC(C/r -> C)               │     │
        └─────────────────────────────┘     │
               │              │              │
               ▼              ▼              │
        ┌─────────────────────────────┐     │
        │           Add               │     │
        └─────────────────────────────┘     │
               │                            │
               ▼                            │
        ┌─────────────┐                     │
        │   Sigmoid   │                     │
        │ (C x 1 x 1) │                     │
        └─────────────┘                     │
               │                            │
               ▼                            │
        ┌─────────────────────────────────────┐
        │              Multiply               │
        │      (C x 1 x 1) × (C x H x W)     │
        └─────────────────────────────────────┘
               │
               ▼
        Output Features (C x H x W)
    
    Attributes:
        avg_pool: Global average pooling layer.
        max_pool: Global max pooling layer.
        mlp: Shared multi-layer perceptron.
        sigmoid: Final activation.
    
    Reference:
        Hu et al., "Squeeze-and-Excitation Networks" (CVPR 2018)
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """
        Initialize Channel Attention module.
        
        Args:
            channels (int): Number of input channels.
                This determines the input/output size of the MLP.
            reduction (int): Reduction ratio for MLP bottleneck.
                Higher reduction = fewer parameters but less capacity.
                Default: 16 (reduces channels by 16x in bottleneck).
        
        Example:
            >>> # 256 channels with 16x reduction (bottleneck = 16)
            >>> channel_attn = ChannelAttention(channels=256, reduction=16)
        """
        super().__init__()
        
        # Global pooling layers - reduce spatial dimensions to 1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (C, H, W) -> (C, 1, 1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # (C, H, W) -> (C, 1, 1)
        
        # Shared MLP for both pooling branches
        # Architecture: C -> C/r -> C (bottleneck design)
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # Reduce
            nn.ReLU(inplace=True),                                    # Activation
            nn.Linear(channels // reduction, channels, bias=False)   # Expand
        )
        
        # Sigmoid for attention weights in [0, 1]
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply channel attention to input feature tensor.
        
        Args:
            x (torch.Tensor): Input features of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Attention-weighted features of same shape.
                Important channels are emphasized, less important suppressed.
        """
        batch_size, channels, _, _ = x.size()
        
        # =====================================================================
        # Branch 1: Average Pooling
        # Captures global average response of each channel
        # =====================================================================
        avg_out = self.avg_pool(x).view(batch_size, channels)  # (B, C)
        avg_out = self.mlp(avg_out)  # (B, C)
        
        # =====================================================================
        # Branch 2: Max Pooling
        # Captures maximum response of each channel
        # =====================================================================
        max_out = self.max_pool(x).view(batch_size, channels)  # (B, C)
        max_out = self.mlp(max_out)  # (B, C)
        
        # =====================================================================
        # Combine branches and compute attention weights
        # =====================================================================
        attention = self.sigmoid(avg_out + max_out)  # (B, C)
        
        # Reshape for broadcasting: (B, C) -> (B, C, 1, 1)
        attention = attention.view(batch_size, channels, 1, 1)
        
        # Apply attention (broadcasts across spatial dimensions)
        return x * attention


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM).
    
    CBAM combines channel and spatial attention sequentially for
    comprehensive feature refinement. It answers both "what" and
    "where" to focus attention.
    
    Why use CBAM for Pool Detection?
        - Pools have distinctive color/texture (channel attention helps)
        - Pools appear at specific locations (spatial attention helps)
        - Suppresses background noise (buildings, roads, vegetation)
    
    Processing Flow:
        Input Features
              │
              ▼
        ┌─────────────────┐
        │ Channel         │  "What to focus on?"
        │ Attention       │  Emphasize pool-related channels
        └─────────────────┘
              │
              ▼
        ┌─────────────────┐
        │ Spatial         │  "Where to focus?"
        │ Attention       │  Emphasize pool locations
        └─────────────────┘
              │
              ▼
        Output Features
    
    Given an intermediate feature map F ∈ R^(C×H×W):
        F' = Mc(F) ⊗ F          (Channel attention)
        F'' = Ms(F') ⊗ F'       (Spatial attention)
    
    Where:
        - Mc: Channel attention map (C × 1 × 1)
        - Ms: Spatial attention map (1 × H × W)
        - ⊗: Element-wise multiplication
    
    Attributes:
        channel_attention: ChannelAttention module.
        spatial_attention: SpatialAttention module.
    
    Reference:
        Woo et al., "CBAM: Convolutional Block Attention Module" (ECCV 2018)
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
            channels (int): Number of input feature channels.
            reduction (int): Channel reduction ratio. Default: 16.
            kernel_size (int): Kernel size for spatial attention. Default: 7.
        
        Example:
            >>> cbam = CBAM(channels=256, reduction=16, kernel_size=7)
            >>> features = torch.randn(2, 256, 32, 32)
            >>> output = cbam(features)  # Shape: (2, 256, 32, 32)
        """
        super().__init__()
        
        # Initialize sub-modules
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply CBAM attention (channel then spatial).
        
        Args:
            x (torch.Tensor): Input features of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Attention-enhanced features of same shape.
        """
        # Step 1: Apply channel attention
        # Emphasizes important feature channels (e.g., blue color for pools)
        x = self.channel_attention(x)
        
        # Step 2: Apply spatial attention
        # Emphasizes important spatial regions (e.g., pool locations)
        x = self.spatial_attention(x)
        
        return x


# =============================================================================
# LOSS FUNCTIONS
# =============================================================================
# Custom loss functions improve training for object detection tasks.
# We implement Focal Loss (class imbalance) and IoU-based losses
# (better bounding box regression).
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance in object detection.
    
    Standard cross-entropy loss treats all examples equally, which is
    problematic when:
        - Most image regions are background (easy negatives)
        - Actual objects (pools) are rare (hard examples)
    
    Focal Loss down-weights easy examples and focuses on hard ones:
        FL(p_t) = -α_t × (1 - p_t)^γ × log(p_t)
    
    Where:
        - p_t: Probability of correct class
        - α_t: Class weighting factor
        - γ (gamma): Focusing parameter (γ ≥ 0)
    
    Effect of γ (gamma):
        - γ = 0: Standard cross-entropy
        - γ = 1: Moderate down-weighting
        - γ = 2: Strong down-weighting (recommended)
    
    Visual Example:
        Cross-Entropy vs Focal Loss
        
        Loss │                        γ = 0 (CE)
             │   ╲
             │    ╲         γ = 1
             │     ╲    ╲
             │      ╲    ╲   γ = 2
             │       ╲    ─────
             │        ─────────
             └──────────────────── p_t
             0                    1
        
        At high p_t (easy examples), Focal Loss is near 0.
        At low p_t (hard examples), Focal Loss behaves like CE.
    
    Attributes:
        alpha (float): Weighting for positive class. Default: 0.25.
        gamma (float): Focusing parameter. Default: 2.0.
        reduction (str): Loss reduction method.
    
    Reference:
        Lin et al., "Focal Loss for Dense Object Detection" (ICCV 2017)
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
            alpha (float): Weighting factor for positive examples.
                Set < 0.5 when positives are common, > 0.5 when rare.
                Default: 0.25 (pools are rare in aerial images).
            gamma (float): Focusing parameter.
                Higher gamma = more focus on hard examples.
                Typical values: 0.5, 1.0, 2.0, 5.0.
                Default: 2.0 (works well for most cases).
            reduction (str): How to reduce batch losses:
                - 'none': No reduction, return per-sample losses
                - 'mean': Average of all losses
                - 'sum': Sum of all losses
        """
        super().__init__()
        
        self.alpha = alpha      # Class weighting
        self.gamma = gamma      # Focusing parameter
        self.reduction = reduction

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute focal loss between predictions and targets.
        
        Focal Loss Formula:
            FL = -α × (1 - p_t)^γ × CE(p, y)
            
            where p_t = p if y=1 else (1-p)
        
        Args:
            predictions (torch.Tensor): Predicted logits.
                Shape: (N, C) for C classes, or (N,) for binary.
            targets (torch.Tensor): Ground truth class indices.
                Shape: (N,) with values in [0, C-1].
        
        Returns:
            torch.Tensor: Focal loss value.
                Scalar if reduction='mean' or 'sum', else (N,).
        """
        # Compute standard cross-entropy (no reduction)
        # CE = -log(p_t) where p_t is probability of true class
        ce_loss = F.cross_entropy(predictions, targets, reduction="none")
        
        # Compute p_t (probability assigned to true class)
        # p_t = exp(-CE) since CE = -log(p_t)
        pt = torch.exp(-ce_loss)
        
        # Apply focal modulation
        # (1 - p_t)^γ reduces loss for well-classified examples
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # 'none'
            return focal_loss


class IoULoss(nn.Module):
    """
    Intersection over Union (IoU) based loss for bounding box regression.
    
    Unlike L1/L2 losses that treat box coordinates independently, IoU-based
    losses directly optimize the overlap between predicted and target boxes.
    This leads to better-aligned bounding boxes.
    
    Supported Loss Types:
    
    1. IoU Loss (basic):
        - Loss = 1 - IoU
        - Simple but has gradient issues when boxes don't overlap
    
    2. GIoU Loss (Generalized IoU):
        - Adds penalty based on area of enclosing box
        - Works even when boxes don't overlap
        - GIoU = IoU - (C - Union) / C
        - Where C is area of smallest enclosing box
    
    3. DIoU Loss (Distance IoU):
        - Adds penalty based on center point distance
        - Faster convergence than GIoU
        - DIoU = IoU - d² / c²
        - Where d = center distance, c = enclosing box diagonal
    
    4. CIoU Loss (Complete IoU):
        - Adds aspect ratio consistency penalty
        - Best for accurate localization
        - CIoU = DIoU - α × v
        - Where v measures aspect ratio difference
    
    IoU Calculation:
        ┌──────────────────────┐
        │    Box 1             │
        │   ┌──────────────┐   │
        │   │ Intersection │   │
        │   │    (I)       │   │
        │   └──────────────┘   │
        │              Box 2   │
        └──────────────────────┘
        
        IoU = I / (Area1 + Area2 - I)
    
    Attributes:
        loss_type (str): Type of IoU loss ('iou', 'giou', 'diou', 'ciou').
    
    Reference:
        - GIoU: Rezatofighi et al., "Generalized Intersection over Union" (CVPR 2019)
        - DIoU/CIoU: Zheng et al., "Distance-IoU Loss" (AAAI 2020)
    """

    def __init__(self, loss_type: str = "ciou") -> None:
        """
        Initialize IoU Loss.
        
        Args:
            loss_type (str): Type of IoU loss to compute.
                - 'iou': Basic IoU loss (1 - IoU)
                - 'giou': Generalized IoU (handles non-overlapping boxes)
                - 'diou': Distance IoU (faster convergence)
                - 'ciou': Complete IoU (best accuracy, default)
        
        Raises:
            ValueError: If loss_type is not recognized.
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
        Compute IoU-based loss between predicted and target boxes.
        
        Args:
            pred_boxes (torch.Tensor): Predicted boxes in xyxy format.
                Shape: (N, 4) where each row is [x1, y1, x2, y2].
            target_boxes (torch.Tensor): Target boxes in xyxy format.
                Shape: (N, 4) where each row is [x1, y1, x2, y2].
        
        Returns:
            torch.Tensor: Mean loss value (scalar).
        """
        # Select loss computation based on type
        if self.loss_type == "iou":
            iou = self._compute_iou(pred_boxes, target_boxes)
            loss = 1 - iou
        elif self.loss_type == "giou":
            giou = self._compute_giou(pred_boxes, target_boxes)
            loss = 1 - giou
        elif self.loss_type == "diou":
            diou = self._compute_diou(pred_boxes, target_boxes)
            loss = 1 - diou
        else:  # ciou
            ciou = self._compute_ciou(pred_boxes, target_boxes)
            loss = 1 - ciou
        
        return loss.mean()

    def _compute_iou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard Intersection over Union.
        
        IoU = Area of Intersection / Area of Union
        
        Args:
            boxes1: First set of boxes (N, 4) in xyxy format.
            boxes2: Second set of boxes (N, 4) in xyxy format.
        
        Returns:
            IoU values for each box pair (N,).
        """
        # =====================================================================
        # Step 1: Compute intersection
        # =====================================================================
        # Intersection top-left corner: max of both top-lefts
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        
        # Intersection bottom-right corner: min of both bottom-rights
        inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        # Intersection area (clamp negative values to 0)
        inter_width = torch.clamp(inter_x2 - inter_x1, min=0)
        inter_height = torch.clamp(inter_y2 - inter_y1, min=0)
        inter_area = inter_width * inter_height
        
        # =====================================================================
        # Step 2: Compute union
        # =====================================================================
        # Individual box areas
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # Union = Area1 + Area2 - Intersection
        union_area = area1 + area2 - inter_area
        
        # =====================================================================
        # Step 3: Compute IoU
        # =====================================================================
        # Add small epsilon to avoid division by zero
        return inter_area / (union_area + 1e-7)

    def _compute_giou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Generalized IoU.
        
        GIoU = IoU - (C - Union) / C
        
        Where C is the area of the smallest enclosing box.
        GIoU penalizes boxes that are far apart even when IoU = 0.
        
        Range: [-1, 1] (unlike IoU which is [0, 1])
        """
        iou = self._compute_iou(boxes1, boxes2)
        
        # Enclosing box (smallest box containing both)
        enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        enc_area = (enc_x2 - enc_x1) * (enc_y2 - enc_y1)
        
        # Compute union (needed for GIoU formula)
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        inter_x1 = torch.max(boxes1[:, 0], boxes2[:, 0])
        inter_y1 = torch.max(boxes1[:, 1], boxes2[:, 1])
        inter_x2 = torch.min(boxes1[:, 2], boxes2[:, 2])
        inter_y2 = torch.min(boxes1[:, 3], boxes2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * \
                     torch.clamp(inter_y2 - inter_y1, min=0)
        
        union_area = area1 + area2 - inter_area
        
        # GIoU = IoU - (Enclosing - Union) / Enclosing
        giou = iou - (enc_area - union_area) / (enc_area + 1e-7)
        
        return giou

    def _compute_diou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Distance IoU.
        
        DIoU = IoU - d² / c²
        
        Where:
        - d: Euclidean distance between box centers
        - c: Diagonal length of smallest enclosing box
        
        DIoU directly minimizes center distance, leading to faster convergence.
        """
        iou = self._compute_iou(boxes1, boxes2)
        
        # Compute center points
        center1_x = (boxes1[:, 0] + boxes1[:, 2]) / 2
        center1_y = (boxes1[:, 1] + boxes1[:, 3]) / 2
        center2_x = (boxes2[:, 0] + boxes2[:, 2]) / 2
        center2_y = (boxes2[:, 1] + boxes2[:, 3]) / 2
        
        # Squared center distance: d²
        center_dist = (center1_x - center2_x) ** 2 + (center1_y - center2_y) ** 2
        
        # Enclosing box diagonal squared: c²
        enc_x1 = torch.min(boxes1[:, 0], boxes2[:, 0])
        enc_y1 = torch.min(boxes1[:, 1], boxes2[:, 1])
        enc_x2 = torch.max(boxes1[:, 2], boxes2[:, 2])
        enc_y2 = torch.max(boxes1[:, 3], boxes2[:, 3])
        
        enc_diag = (enc_x2 - enc_x1) ** 2 + (enc_y2 - enc_y1) ** 2
        
        # DIoU = IoU - d² / c²
        diou = iou - center_dist / (enc_diag + 1e-7)
        
        return diou

    def _compute_ciou(
        self,
        boxes1: torch.Tensor,
        boxes2: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Complete IoU (recommended for detection).
        
        CIoU = DIoU - α × v
        
        Where:
        - v: Measures aspect ratio consistency
        - α: Trade-off parameter (computed automatically)
        
        v = (4/π²) × (arctan(w_gt/h_gt) - arctan(w_pred/h_pred))²
        α = v / (1 - IoU + v)
        
        CIoU considers overlap, center distance, AND aspect ratio.
        """
        import math
        
        # Start with DIoU
        diou = self._compute_diou(boxes1, boxes2)
        
        # Compute aspect ratio terms
        w1 = boxes1[:, 2] - boxes1[:, 0]  # Width of box 1
        h1 = boxes1[:, 3] - boxes1[:, 1]  # Height of box 1
        w2 = boxes2[:, 2] - boxes2[:, 0]  # Width of box 2
        h2 = boxes2[:, 3] - boxes2[:, 1]  # Height of box 2
        
        # v: Aspect ratio consistency term
        # v = (4/π²) × (arctan(w2/h2) - arctan(w1/h1))²
        v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan(w2 / (h2 + 1e-7)) - torch.atan(w1 / (h1 + 1e-7)), 2
        )
        
        # α: Trade-off parameter
        iou = self._compute_iou(boxes1, boxes2)
        alpha = v / (1 - iou + v + 1e-7)
        
        # CIoU = DIoU - α × v
        ciou = diou - alpha * v
        
        return ciou


# =============================================================================
# AUXILIARY MODULES
# =============================================================================

class PoolShapeClassifier(nn.Module):
    """
    Auxiliary classifier for pool shape prediction.
    
    This optional head classifies detected pools into shape categories:
    - Rectangular: Standard rectangular/square pools
    - Oval: Round or oval-shaped pools
    - Irregular: Free-form or L-shaped pools
    
    Can be added as auxiliary head during training for multi-task learning.
    
    Architecture:
        Feature Map (C x H x W)
              │
              ▼
        ┌─────────────────┐
        │ AdaptiveAvgPool │  Global pooling
        │    (1 x 1)      │
        └─────────────────┘
              │
              ▼
        ┌─────────────────┐
        │    Flatten      │
        └─────────────────┘
              │
              ▼
        ┌─────────────────┐
        │   FC (C → 128)  │
        │      ReLU       │
        │   Dropout(0.5)  │
        │  FC (128 → 3)   │
        └─────────────────┘
              │
              ▼
        Shape Logits (3,)
    
    Attributes:
        classifier: Sequential classification head.
        class_names: List of shape category names.
    """

    def __init__(
        self,
        input_features: int = 256,
        num_classes: int = 3
    ) -> None:
        """
        Initialize Pool Shape Classifier.
        
        Args:
            input_features (int): Number of input channels from backbone.
                Should match the feature map channel count.
            num_classes (int): Number of shape categories.
                Default: 3 (rectangular, oval, irregular).
        """
        super().__init__()
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),         # Global average pooling
            nn.Flatten(),                     # (B, C, 1, 1) -> (B, C)
            nn.Linear(input_features, 128),   # Reduce dimensions
            nn.ReLU(inplace=True),            # Activation
            nn.Dropout(0.5),                  # Regularization
            nn.Linear(128, num_classes)       # Output logits
        )
        
        # Human-readable class names
        self.class_names = ["rectangular", "oval", "irregular"]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Classify pool shape from feature map.
        
        Args:
            x (torch.Tensor): Feature map of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Shape logits of shape (B, num_classes).
                Use softmax for probabilities.
        """
        return self.classifier(x)


# =============================================================================
# TESTING AND DEMONSTRATION
# =============================================================================

def main() -> None:
    """
    Test and demonstrate custom layers.
    
    This function creates sample inputs and runs each custom layer
    to verify correct operation and output shapes.
    """
    logger.info("Testing custom layers...")
    
    # =========================================================================
    # Test CBAM Attention
    # =========================================================================
    cbam = CBAM(channels=64, reduction=16, kernel_size=7)
    x = torch.randn(2, 64, 32, 32)  # Batch=2, Channels=64, H=W=32
    out = cbam(x)
    logger.info(f"CBAM: input {x.shape} -> output {out.shape}")
    assert x.shape == out.shape, "CBAM should preserve shape"
    
    # =========================================================================
    # Test Focal Loss
    # =========================================================================
    focal = FocalLoss(alpha=0.25, gamma=2.0)
    preds = torch.randn(10, 2)    # 10 samples, 2 classes
    targets = torch.randint(0, 2, (10,))  # Random labels
    loss = focal(preds, targets)
    logger.info(f"Focal Loss: {loss.item():.4f}")
    
    # =========================================================================
    # Test IoU Loss variants
    # =========================================================================
    for loss_type in ["iou", "giou", "diou", "ciou"]:
        iou_loss = IoULoss(loss_type=loss_type)
        boxes1 = torch.tensor([[10, 10, 50, 50], [20, 20, 60, 60]], dtype=torch.float32)
        boxes2 = torch.tensor([[15, 15, 55, 55], [25, 25, 65, 65]], dtype=torch.float32)
        loss = iou_loss(boxes1, boxes2)
        logger.info(f"{loss_type.upper()} Loss: {loss.item():.4f}")
    
    # =========================================================================
    # Test Shape Classifier
    # =========================================================================
    classifier = PoolShapeClassifier(input_features=256, num_classes=3)
    features = torch.randn(4, 256, 16, 16)  # 4 pool features
    logits = classifier(features)
    logger.info(f"Shape Classifier: {features.shape} -> {logits.shape}")
    
    logger.info("All tests passed!")


if __name__ == "__main__":
    main()
