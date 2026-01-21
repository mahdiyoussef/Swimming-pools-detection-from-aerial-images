"""
Model Validation Script for Swimming Pool Detection System.

Author: Swimming Pool Detection Team
Date: 2026-01-02
Updated: 2026-01-21 - Migrated from YOLOv11 to YOLO26
"""

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


class Validator:
    """Model validation for YOLO26 swimming pool detection."""

    def __init__(
        self,
        model_path: str,
        data_yaml: Optional[str] = None,
        output_dir: Optional[str] = None
    ) -> None:
        self.project_root = Path(__file__).resolve().parent.parent
        self.model_path = Path(model_path)
        
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        if data_yaml is None:
            data_yaml = self.project_root / "data" / "dataset.yaml"
        self.data_yaml = Path(data_yaml)
        
        if output_dir is None:
            output_dir = self.project_root / "logs" / "validation"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.model = self._load_model()

    def _load_model(self) -> "YOLO":
        from ultralytics import YOLO
        logger.info(f"Loading model from: {self.model_path}")
        return YOLO(str(self.model_path))

    def validate(
        self,
        split: str = "val",
        conf_threshold: float = 0.001,
        iou_threshold: float = 0.7,
        batch_size: int = 16,
        image_size: int = 640,
        device: str = "0"
    ) -> Dict[str, Any]:
        """Run validation and compute metrics."""
        logger.info(f"Running Validation on {split} split")
        
        if not self.data_yaml.exists():
            raise FileNotFoundError(f"Dataset config not found: {self.data_yaml}")
        
        start_time = time.time()
        
        results = self.model.val(
            data=str(self.data_yaml),
            split=split,
            batch=batch_size,
            imgsz=image_size,
            conf=conf_threshold,
            iou=iou_threshold,
            device=device,
            plots=True,
            save_json=True
        )
        
        total_time = time.time() - start_time
        metrics = self._extract_metrics(results, total_time)
        self._log_metrics(metrics)
        self._save_metrics(metrics, split)
        
        return metrics

    def _extract_metrics(self, results: Any, total_time: float) -> Dict[str, Any]:
        metrics = {"total_time_seconds": total_time}
        
        if hasattr(results, "box"):
            box = results.box
            metrics.update({
                "mAP50": float(box.map50) if hasattr(box, "map50") else None,
                "mAP50-95": float(box.map) if hasattr(box, "map") else None,
                "precision": float(box.mp) if hasattr(box, "mp") else None,
                "recall": float(box.mr) if hasattr(box, "mr") else None,
            })
        
        if metrics.get("precision") and metrics.get("recall"):
            p, r = metrics["precision"], metrics["recall"]
            metrics["f1_score"] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        
        return metrics

    def _log_metrics(self, metrics: Dict[str, Any]) -> None:
        logger.info("Validation Results:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")

    def _save_metrics(self, metrics: Dict[str, Any], split: str) -> None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"validation_{split}_{timestamp}.json"
        
        with open(output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"Metrics saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate YOLO26 model")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", type=str, default="0")
    
    args = parser.parse_args()
    
    validator = Validator(model_path=args.model, data_yaml=args.data)
    validator.validate(split=args.split, batch_size=args.batch_size, device=args.device)


if __name__ == "__main__":
    main()
