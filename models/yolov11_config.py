"""
YOLOv11 Model Configuration Module for Swimming Pool Detection System.

This module provides configuration and initialization utilities
for YOLOv11 models used in swimming pool detection.

Author: Swimming Pool Detection Team
Date: 2026-01-02
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


# Model variant configurations
MODEL_VARIANTS = {
    "yolov11n": {
        "name": "yolo11n.pt",
        "description": "Nano - Fastest, lowest accuracy",
        "parameters": "2.6M",
        "flops": "6.5G"
    },
    "yolov11s": {
        "name": "yolo11s.pt",
        "description": "Small - Fast with good accuracy",
        "parameters": "9.4M",
        "flops": "21.5G"
    },
    "yolov11m": {
        "name": "yolo11m.pt",
        "description": "Medium - Balanced speed and accuracy",
        "parameters": "20.1M",
        "flops": "68.0G"
    },
    "yolov11l": {
        "name": "yolo11l.pt",
        "description": "Large - High accuracy, slower",
        "parameters": "25.3M",
        "flops": "86.9G"
    },
    "yolov11x": {
        "name": "yolo11x.pt",
        "description": "Extra Large - Highest accuracy",
        "parameters": "56.9M",
        "flops": "194.9G"
    }
}


class YOLOv11Config:
    """
    Configuration class for YOLOv11 models.

    This class handles model configuration, variant selection,
    and initialization parameters for YOLOv11 training and inference.

    Attributes:
        variant: Model variant name (yolov11n/s/m/l/x).
        num_classes: Number of detection classes.
        pretrained: Whether to use pretrained weights.
        pretrained_weights: Path to custom pretrained weights.
    """

    def __init__(
        self,
        variant: str = "yolov11n",
        num_classes: int = 1,
        pretrained: bool = True,
        pretrained_weights: Optional[str] = None,
        config_path: Optional[str] = None
    ) -> None:
        """
        Initialize YOLOv11 configuration.

        Args:
            variant: Model variant (yolov11n, yolov11s, yolov11m, yolov11l, yolov11x).
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
        
        logger.info(f"Configured YOLOv11 variant: {self.variant}")
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
        
        self.variant = model_config.get("variant", "yolov11n")
        self.num_classes = model_config.get("num_classes", 1)
        self.pretrained = model_config.get("pretrained", True)
        self.pretrained_weights = model_config.get("pretrained_weights", None)

    def get_model_name(self) -> str:
        """
        Get the model filename for loading.

        Returns:
            str: Model filename (e.g., 'yolo11n.pt').
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
        Create and return a YOLOv11 model instance.

        Returns:
            YOLO: Initialized YOLOv11 model.

        Raises:
            ImportError: If ultralytics is not installed.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
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

        Args:
            conf_threshold: Confidence threshold for detections.
            iou_threshold: IOU threshold for NMS.
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
        print("\nAvailable YOLOv11 Variants:")
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
            "Install with: pip install ultralytics"
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
        description="YOLOv11 model configuration utility"
    )
    parser.add_argument(
        "--list-variants",
        action="store_true",
        help="List all available model variants"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="yolov11n",
        help="Model variant to display info for"
    )
    
    args = parser.parse_args()
    
    if args.list_variants:
        YOLOv11Config.list_variants()
    else:
        config = YOLOv11Config(variant=args.variant)
        info = config.get_model_info()
        print(f"\n{args.variant} Configuration:")
        print(f"  Model file:   {config.get_model_name()}")
        print(f"  Description:  {info['description']}")
        print(f"  Parameters:   {info['parameters']}")
        print(f"  FLOPs:        {info['flops']}")


if __name__ == "__main__":
    main()
