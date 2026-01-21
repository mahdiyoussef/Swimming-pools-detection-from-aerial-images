"""
YOLO26 Model Configuration Module for Swimming Pool Detection/Segmentation System.

This module provides configuration and initialization utilities
for YOLO26 models used in swimming pool detection and instance segmentation.

YOLO26 Key Features:
    - NMS-free end-to-end inference for reduced latency
    - MuSGD optimizer for stable training and faster convergence
    - 43% faster CPU inference compared to YOLO11
    - Progressive Loss Balancing (ProgLoss) for improved accuracy
    - Instance segmentation support with polygon mask output

Author: Swimming Pool Detection Team
Date: 2026-01-02
Updated: 2026-01-21 - Added instance segmentation support with YOLO26-seg
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# Model variant configurations - Detection models
MODEL_VARIANTS = {
    # Detection models
    "yolo26n": {
        "name": "yolo26n.pt",
        "task": "detect",
        "description": "Nano Detection - Fastest, optimized for edge devices",
        "parameters": "~2.5M",
        "flops": "~6G"
    },
    "yolo26s": {
        "name": "yolo26s.pt",
        "task": "detect",
        "description": "Small Detection - Fast with good accuracy",
        "parameters": "~9M",
        "flops": "~20G"
    },
    "yolo26m": {
        "name": "yolo26m.pt",
        "task": "detect",
        "description": "Medium Detection - Balanced speed and accuracy",
        "parameters": "~20M",
        "flops": "~65G"
    },
    "yolo26l": {
        "name": "yolo26l.pt",
        "task": "detect",
        "description": "Large Detection - High accuracy, slower",
        "parameters": "~25M",
        "flops": "~85G"
    },
    "yolo26x": {
        "name": "yolo26x.pt",
        "task": "detect",
        "description": "Extra Large Detection - Highest accuracy",
        "parameters": "~55M",
        "flops": "~190G"
    },
    # Segmentation models - Output polygon masks for precise contours
    "yolo26n-seg": {
        "name": "yolo26n-seg.pt",
        "task": "segment",
        "description": "Nano Segmentation - Fastest, polygon mask output",
        "parameters": "~2.8M",
        "flops": "~8G"
    },
    "yolo26s-seg": {
        "name": "yolo26s-seg.pt",
        "task": "segment",
        "description": "Small Segmentation - Fast with polygon masks (recommended)",
        "parameters": "~11M",
        "flops": "~25G"
    },
    "yolo26m-seg": {
        "name": "yolo26m-seg.pt",
        "task": "segment",
        "description": "Medium Segmentation - Balanced with polygon masks",
        "parameters": "~23M",
        "flops": "~75G"
    },
    "yolo26l-seg": {
        "name": "yolo26l-seg.pt",
        "task": "segment",
        "description": "Large Segmentation - High accuracy polygon masks",
        "parameters": "~28M",
        "flops": "~95G"
    },
    "yolo26x-seg": {
        "name": "yolo26x-seg.pt",
        "task": "segment",
        "description": "Extra Large Segmentation - Best mask quality",
        "parameters": "~60M",
        "flops": "~210G"
    }
}


class YOLO26Config:
    """
    Configuration class for YOLO26 models (detection and segmentation).

    This class handles model configuration, variant selection,
    and initialization parameters for YOLO26 training and inference.
    Supports both detection (bounding boxes) and segmentation (polygon masks).

    Attributes:
        variant: Model variant name (yolo26n/s/m/l/x or yolo26n-seg/s-seg/m-seg/l-seg/x-seg).
        task: Task type ('detect' or 'segment').
        num_classes: Number of detection/segmentation classes.
        pretrained: Whether to use pretrained weights.
        pretrained_weights: Path to custom pretrained weights.
    """

    def __init__(
        self,
        variant: str = "yolo26n",
        num_classes: int = 1,
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None,
        config_path: Optional[str] = None
    ) -> None:
        """
        Initialize YOLO26 configuration.

        Args:
            variant: Model variant (yolo26n, yolo26s, yolo26m, yolo26l, yolo26x).
            num_classes: Number of classes to detect.
            pretrained: Whether to load pretrained weights.
            pretrained_weights: Path to custom pretrained weights file.
            config_path: Path to configuration YAML file (overrides other args).
        """
        # Load from config file if provided
        if config_path is not None:
            self._load_from_config(config_path)
        else:
            self.variant = variant
            self.num_classes = num_classes
            self.pretrained = pretrained
            self.pretrained_weights = pretrained_weights
        
        # Validate variant
        if self.variant not in MODEL_VARIANTS:
            valid_variants = list(MODEL_VARIANTS.keys())
            raise ValueError(
                f"Invalid model variant: {self.variant}. "
                f"Choose from: {valid_variants}"
            )
        
        logger.info(f"Configured YOLO26 variant: {self.variant}")
        logger.info(f"Number of classes: {self.num_classes}")

    def _load_from_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        model_config = config.get("model", {})
        
        self.variant = model_config.get("variant", "yolo26n")
        self.num_classes = model_config.get("num_classes", 1)
        self.pretrained = model_config.get("pretrained", True)
        self.pretrained_weights = model_config.get("pretrained_weights", None)

    def get_model_name(self) -> str:
        """
        Get the model filename for loading.

        Returns:
            str: Model filename (e.g., 'yolo26n.pt').
        """
        return MODEL_VARIANTS[self.variant]["name"]

    def get_model_info(self) -> Dict[str, str]:
        """
        Get information about the selected model variant.

        Returns:
            Dict containing model description, parameters, and FLOPs.
        """
        return MODEL_VARIANTS[self.variant]

    def create_model(self) -> "YOLO":
        """
        Create and return a YOLO26 model instance.

        Returns:
            YOLO: Initialized YOLO26 model.

        Raises:
            ImportError: If ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics>=8.4.0"
            )
        
        # Determine model source
        if self.pretrained_weights and Path(self.pretrained_weights).exists():
            model_source = self.pretrained_weights
            logger.info(f"Loading custom weights from: {model_source}")
        elif self.pretrained:
            model_source = self.get_model_name()
            logger.info(f"Loading pretrained model: {model_source}")
        else:
            # For training from scratch, still start with pretrained for architecture
            model_source = self.get_model_name()
            logger.info(f"Initializing model architecture from: {model_source}")
        
        model = YOLO(model_source)
        logger.info("Model created successfully")
        
        return model

    def get_training_args(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get training arguments for the model.

        Args:
            data_yaml: Path to dataset.yaml file.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            image_size: Input image size.
            **kwargs: Additional training arguments.

        Returns:
            Dict containing all training arguments.
        """
        training_args = {
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": image_size,
            "patience": kwargs.get("patience", 20),
            "save_period": kwargs.get("save_period", 10),
            "device": kwargs.get("device", "0"),
            "workers": kwargs.get("workers", 8),
            "project": kwargs.get("project", "logs"),
            "name": kwargs.get("name", "train"),
            "exist_ok": kwargs.get("exist_ok", True),
            "pretrained": self.pretrained,
            "optimizer": kwargs.get("optimizer", "AdamW"),
            "lr0": kwargs.get("learning_rate", 0.001),
            "weight_decay": kwargs.get("weight_decay", 0.0005),
            "warmup_epochs": kwargs.get("warmup_epochs", 3),
            "cos_lr": kwargs.get("cos_lr", True),
            "amp": kwargs.get("amp", True),
            "verbose": kwargs.get("verbose", True),
        }
        
        return training_args

    def get_inference_args(
        self,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        Get inference arguments for the model.

        Note: YOLO26 uses NMS-free end-to-end inference, but the
        iou_threshold is kept for API compatibility.

        Args:
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IOU threshold (kept for compatibility).
            image_size: Input image size.
            **kwargs: Additional inference arguments.

        Returns:
            Dict containing all inference arguments.
        """
        inference_args = {
            "conf": conf_threshold,
            "iou": iou_threshold,
            "imgsz": image_size,
            "device": kwargs.get("device", "0"),
            "max_det": kwargs.get("max_det", 300),
            "classes": kwargs.get("classes", None),
            "verbose": kwargs.get("verbose", False),
        }
        
        return inference_args

    @staticmethod
    def list_variants() -> None:
        """Print information about all available model variants."""
        print("\nAvailable YOLO26 Variants:")
        print("-" * 60)
        for variant, info in MODEL_VARIANTS.items():
            print(f"\n{variant}:")
            print(f"  Description: {info['description']}")
            print(f"  Parameters:  {info['parameters']}")
            print(f"  FLOPs:       {info['flops']}")
        print("-" * 60)


def load_model_from_checkpoint(
    checkpoint_path: Union[str, Path],
    device: str = "0"
) -> "YOLO":
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to the model checkpoint (.pt file).
        device: Device to load the model on.

    Returns:
        YOLO: Loaded model ready for inference.

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist.
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        raise ImportError(
            "ultralytics package not installed. "
            "Install with: pip install ultralytics>=8.4.0"
        )
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading model from checkpoint: {checkpoint_path}")
    model = YOLO(str(checkpoint_path))
    
    return model


def main() -> None:
    """Main entry point for model configuration display."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YOLO26 model configuration utility"
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List all available model variants"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="yolo26n",
        help="Model variant to display info for"
    )
    
    args = parser.parse_args()
    
    if args.list_variants:
        YOLO26Config.list_variants()
    else:
        config = YOLO26Config(variant=args.variant)
        info = config.get_model_info()
        print(f"\n{args.variant} Configuration:")
        print(f"  Model file:   {config.get_model_name()}")
        print(f"  Description:  {info['description']}")
        print(f"  Parameters:   {info['parameters']}")
        print(f"  FLOPs:        {info['flops']}")


if __name__ == "__main__":
    main()
