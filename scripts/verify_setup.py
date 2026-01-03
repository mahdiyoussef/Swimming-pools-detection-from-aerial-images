"""
Setup Verification Script for Swimming Pool Detection System.

This module verifies that all dependencies are correctly installed and
the project structure is properly configured.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import importlib
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


# Required packages and their import names
REQUIRED_PACKAGES: Dict[str, str] = {
    "ultralytics": "ultralytics",
    "torch": "torch",
    "torchvision": "torchvision",
    "cv2": "opencv-python",
    "numpy": "numpy",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "tensorboard": "tensorboard",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "pandas": "pandas",
    "sklearn": "scikit-learn",
    "tqdm": "tqdm",
    "albumentations": "albumentations",
}

OPTIONAL_PACKAGES: Dict[str, str] = {
    "kaggle": "kaggle",
    "roboflow": "roboflow",
}


def check_python_version() -> Tuple[bool, str]:
    """
    Check if Python version meets requirements.

    Returns:
        Tuple[bool, str]: Success status and message.
    """
    version_info = sys.version_info
    if version_info.major == 3 and version_info.minor >= 8:
        return True, f"Python {version_info.major}.{version_info.minor}.{version_info.micro}"
    return False, f"Python 3.8+ required, found {version_info.major}.{version_info.minor}"


def check_cuda_availability() -> Tuple[bool, str]:
    """
    Check if CUDA is available for GPU acceleration.

    Returns:
        Tuple[bool, str]: Availability status and message.
    """
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            cuda_version = torch.version.cuda
            return True, f"CUDA {cuda_version} - {device_name}"
        return False, "CUDA not available, CPU mode will be used"
    except ImportError:
        return False, "PyTorch not installed, cannot check CUDA"


def check_package(import_name: str) -> Tuple[bool, str]:
    """
    Check if a package is installed and get its version.

    Args:
        import_name: The import name of the package.

    Returns:
        Tuple[bool, str]: Installation status and version or error message.
    """
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "unknown")
        return True, version
    except ImportError:
        return False, "Not installed"


def check_directory_structure(project_root: Path) -> List[str]:
    """
    Check if required directories exist.

    Args:
        project_root: The root path of the project.

    Returns:
        List[str]: List of missing directories.
    """
    required_dirs = [
        "config",
        "data/raw",
        "data/processed",
        "data/splits/train",
        "data/splits/val",
        "data/splits/test",
        "datasets",
        "preprocessing",
        "models",
        "training",
        "inference",
        "logs/tensorboard",
        "logs/checkpoints",
        "tests",
        "scripts",
        "docs",
    ]
    
    missing = []
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if not full_path.exists():
            missing.append(dir_path)
    
    return missing


def check_config_files(project_root: Path) -> List[str]:
    """
    Check if required configuration files exist.

    Args:
        project_root: The root path of the project.

    Returns:
        List[str]: List of missing configuration files.
    """
    required_files = [
        "config/dataset_config.yaml",
        "config/training_config.yaml",
        "requirements.txt",
    ]
    
    missing = []
    for file_path in required_files:
        full_path = project_root / file_path
        if not full_path.exists():
            missing.append(file_path)
    
    return missing


def print_section(title: str) -> None:
    """Print a formatted section header."""
    logger.info("=" * 60)
    logger.info(f"  {title}")
    logger.info("=" * 60)


def main() -> None:
    """Main entry point for setup verification."""
    # Get project root
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    all_passed = True
    
    # Check Python version
    print_section("Python Version")
    success, message = check_python_version()
    status = "PASS" if success else "FAIL"
    logger.info(f"  [{status}] {message}")
    all_passed = all_passed and success
    
    # Check CUDA
    print_section("CUDA Availability")
    success, message = check_cuda_availability()
    status = "PASS" if success else "INFO"
    logger.info(f"  [{status}] {message}")
    
    # Check required packages
    print_section("Required Packages")
    for import_name, package_name in REQUIRED_PACKAGES.items():
        success, version = check_package(import_name)
        status = "PASS" if success else "FAIL"
        logger.info(f"  [{status}] {package_name}: {version}")
        all_passed = all_passed and success
    
    # Check optional packages
    print_section("Optional Packages")
    for import_name, package_name in OPTIONAL_PACKAGES.items():
        success, version = check_package(import_name)
        status = "PASS" if success else "WARN"
        logger.info(f"  [{status}] {package_name}: {version}")
    
    # Check directory structure
    print_section("Directory Structure")
    missing_dirs = check_directory_structure(project_root)
    if missing_dirs:
        for dir_path in missing_dirs:
            logger.info(f"  [WARN] Missing: {dir_path}")
    else:
        logger.info("  [PASS] All required directories exist")
    
    # Check configuration files
    print_section("Configuration Files")
    missing_files = check_config_files(project_root)
    if missing_files:
        for file_path in missing_files:
            logger.info(f"  [FAIL] Missing: {file_path}")
            all_passed = False
    else:
        logger.info("  [PASS] All configuration files exist")
    
    # Final summary
    print_section("Verification Summary")
    if all_passed:
        logger.info("  [SUCCESS] All checks passed. Setup is complete.")
        sys.exit(0)
    else:
        logger.info("  [FAILURE] Some checks failed. Please review and fix issues.")
        sys.exit(1)


if __name__ == "__main__":
    main()
