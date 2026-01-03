"""
Training Script for Swimming Pool Detection System.

This module handles the complete training pipeline for YOLOv11-based
swimming pool detection in aerial imagery.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class Trainer:
    """
    Training orchestrator for YOLOv11 swimming pool detection.

    This class handles the complete training workflow including
    configuration loading, model initialization, training execution,
    and checkpoint management.

    Attributes:
        config: Training configuration dictionary.
        project_root: Root path of the project.
        model: YOLOv11 model instance.
    """

    def __init__(self, config_path: str) -> None:
        """
        Initialize the Trainer.

        Args:
            config_path: Path to the training configuration file.
        """
        self.project_root = Path(__file__).resolve().parent.parent
        self.config = self._load_config(config_path)
        self.model = None
        self.training_run_name = self._generate_run_name()
        
        # Setup logging to file
        self._setup_file_logging()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load training configuration from YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            dict: Configuration dictionary.

        Raises:
            FileNotFoundError: If config file doesn't exist.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        
        logger.info(f"Loaded configuration from: {config_path}")
        return config

    def _generate_run_name(self) -> str:
        """Generate a unique name for this training run."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_variant = self.config["model"]["variant"]
        return f"{model_variant}_{timestamp}"

    def _setup_file_logging(self) -> None:
        """Setup logging to file in addition to console."""
        log_dir = self.project_root / self.config["logging"]["log_dir"] / "training"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        log_file = log_dir / f"train_{self.training_run_name}.log"
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        )
        
        logging.getLogger().addHandler(file_handler)
        logger.info(f"Logging to file: {log_file}")

    def _initialize_model(self, model_variant: Optional[str] = None) -> None:
        """
        Initialize the YOLOv11 model.

        Args:
            model_variant: Override model variant from config.
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError(
                "ultralytics package not installed. "
                "Install with: pip install ultralytics"
            )
        
        if model_variant is None:
            model_variant = self.config["model"]["variant"]
        
        # Map variant to model file
        variant_mapping = {
            "yolov11n": "yolo11n.pt",
            "yolov11s": "yolo11s.pt",
            "yolov11m": "yolo11m.pt",
            "yolov11l": "yolo11l.pt",
            "yolov11x": "yolo11x.pt",
        }
        
        model_file = variant_mapping.get(model_variant)
        if model_file is None:
            raise ValueError(f"Unknown model variant: {model_variant}")
        
        # Check for custom pretrained weights
        pretrained_weights = self.config["model"].get("pretrained_weights")
        if pretrained_weights and Path(pretrained_weights).exists():
            model_file = pretrained_weights
            logger.info(f"Using custom pretrained weights: {model_file}")
        else:
            logger.info(f"Loading pretrained model: {model_file}")
        
        self.model = YOLO(model_file)
        logger.info("Model initialized successfully")

    def _get_training_args(self, **overrides: Any) -> Dict[str, Any]:
        """
        Build training arguments from config with optional overrides.

        Args:
            **overrides: Override specific training arguments.

        Returns:
            dict: Complete training arguments.
        """
        training_config = self.config["training"]
        optimizer_config = self.config["optimizer"]
        early_stopping_config = self.config["early_stopping"]
        checkpoint_config = self.config["checkpoint"]
        paths_config = self.config["paths"]
        device_config = self.config["device"]
        
        args = {
            # Data
            "data": str(self.project_root / paths_config["dataset_yaml"]),
            
            # Training
            "epochs": training_config["epochs"],
            "batch": training_config["batch_size"],
            "imgsz": training_config["image_size"],
            "workers": training_config["workers"],
            
            # Optimizer
            "optimizer": optimizer_config["name"],
            "lr0": optimizer_config["learning_rate"],
            "weight_decay": optimizer_config["weight_decay"],
            "momentum": optimizer_config["momentum"],
            "warmup_epochs": optimizer_config["warmup_epochs"],
            "warmup_momentum": optimizer_config["warmup_momentum"],
            "warmup_bias_lr": optimizer_config["warmup_bias_lr"],
            
            # Early stopping
            "patience": early_stopping_config["patience"],
            
            # Checkpointing
            "save_period": checkpoint_config["save_period"],
            "save": checkpoint_config["save_best"],
            
            # Device
            "device": device_config["device_id"] if device_config["cuda"] else "cpu",
            "amp": device_config["amp"],
            
            # Output
            "project": str(self.project_root / paths_config["output_dir"]),
            "name": self.training_run_name,
            "exist_ok": True,
            
            # Logging
            "verbose": self.config["logging"]["verbose"],
            "plots": True,
        }
        
        # Apply cosine LR schedule if configured
        scheduler_config = self.config.get("scheduler", {})
        if scheduler_config.get("name") == "cosine":
            args["cos_lr"] = True
        
        # Apply overrides
        args.update(overrides)
        
        return args

    def train(
        self,
        model_variant: Optional[str] = None,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        resume: Optional[str] = None,
        **kwargs: Any
    ) -> None:
        """
        Execute training.

        Args:
            model_variant: Override model variant from config.
            epochs: Override number of epochs.
            batch_size: Override batch size.
            resume: Path to checkpoint for resuming training.
            **kwargs: Additional training arguments.
        """
        logger.info("=" * 60)
        logger.info("Starting Training")
        logger.info("=" * 60)
        
        # Initialize model
        if resume:
            logger.info(f"Resuming training from: {resume}")
            from ultralytics import YOLO
            self.model = YOLO(resume)
        else:
            self._initialize_model(model_variant)
        
        # Build training arguments
        overrides = {}
        if epochs is not None:
            overrides["epochs"] = epochs
        if batch_size is not None:
            overrides["batch"] = batch_size
        overrides.update(kwargs)
        
        training_args = self._get_training_args(**overrides)
        
        # Log training configuration
        logger.info("Training Configuration:")
        for key, value in training_args.items():
            logger.info(f"  {key}: {value}")
        
        # Verify dataset exists
        data_path = Path(training_args["data"])
        if not data_path.exists():
            raise FileNotFoundError(
                f"Dataset configuration not found: {data_path}. "
                "Please run preprocessing/create_splits.py first."
            )
        
        # Execute training
        logger.info("Starting training loop...")
        
        try:
            results = self.model.train(**training_args)
            
            logger.info("=" * 60)
            logger.info("Training Complete")
            logger.info("=" * 60)
            
            # Log final results
            if results:
                self._log_results(results)
            
        except KeyboardInterrupt:
            logger.warning("Training interrupted by user")
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise

    def _log_results(self, results: Any) -> None:
        """
        Log training results.

        Args:
            results: Training results object.
        """
        logger.info("Training Results:")
        
        if hasattr(results, "box"):
            metrics = results.box
            # Use Metric object attributes instead of .get()
            try:
                logger.info(f"  mAP50: {metrics.map50:.4f}")
                logger.info(f"  mAP50-95: {metrics.map:.4f}")
                logger.info(f"  Precision: {metrics.mp:.4f}")
                logger.info(f"  Recall: {metrics.mr:.4f}")
            except Exception as e:
                logger.warning(f"Could not extract metrics: {e}")
        
        # Log best model path
        output_dir = (
            self.project_root / 
            self.config["paths"]["output_dir"] / 
            self.training_run_name
        )
        best_model = output_dir / "weights" / "best.pt"
        last_model = output_dir / "weights" / "last.pt"
        
        if best_model.exists():
            logger.info(f"Best model saved to: {best_model}")
        if last_model.exists():
            logger.info(f"Last model saved to: {last_model}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train YOLOv11 for swimming pool detection",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to training configuration file"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["yolov11n", "yolov11s", "yolov11m", "yolov11l", "yolov11x"],
        help="YOLOv11 model variant (overrides config)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size for training (overrides config)"
    )
    
    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Input image size (overrides config)"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint for resuming training"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="CUDA device(s) or 'cpu'"
    )
    
    return parser.parse_args()


def main() -> None:
    """Main entry point for training."""
    args = parse_arguments()
    
    # Build overrides from command line arguments
    overrides = {}
    if args.img_size is not None:
        overrides["imgsz"] = args.img_size
    if args.device is not None:
        overrides["device"] = args.device
    
    # Initialize trainer
    trainer = Trainer(config_path=args.config)
    
    # Execute training
    trainer.train(
        model_variant=args.model,
        epochs=args.epochs,
        batch_size=args.batch_size,
        resume=args.resume,
        **overrides
    )


if __name__ == "__main__":
    main()
