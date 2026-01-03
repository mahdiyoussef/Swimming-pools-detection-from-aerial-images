"""
Directory Structure Creation Script for Swimming Pool Detection System.

This module creates the complete directory structure required for the
swimming pool detection project.

Author: Swimming Pool Detection Team
Date: 2026-01-02
"""

import logging
from pathlib import Path
from typing import List

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def get_directory_structure() -> List[Path]:
    """
    Define the complete directory structure for the project.

    Returns:
        List[Path]: List of directory paths to create.
    """
    base_dirs = [
        # Data directories
        Path("data/raw"),
        Path("data/processed"),
        Path("data/annotations"),
        Path("data/splits/train/images"),
        Path("data/splits/train/labels"),
        Path("data/splits/val/images"),
        Path("data/splits/val/labels"),
        Path("data/splits/test/images"),
        Path("data/splits/test/labels"),
        # Log directories
        Path("logs/tensorboard"),
        Path("logs/training"),
        Path("logs/validation"),
        Path("logs/checkpoints"),
        # Output directory
        Path("output"),
        # Model weights directory
        Path("weights"),
    ]
    return base_dirs


def create_directories(base_path: Path) -> None:
    """
    Create all required directories for the project.

    Args:
        base_path: The base path where directories will be created.

    Raises:
        OSError: If directory creation fails due to permission issues.
    """
    directories = get_directory_structure()
    
    for directory in directories:
        full_path = base_path / directory
        try:
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Created directory: {full_path}")
        except OSError as e:
            logger.error(f"Failed to create directory {full_path}: {e}")
            raise


def create_gitkeep_files(base_path: Path) -> None:
    """
    Create .gitkeep files in empty directories to track them in git.

    Args:
        base_path: The base path of the project.
    """
    directories = get_directory_structure()
    
    for directory in directories:
        gitkeep_path = base_path / directory / ".gitkeep"
        try:
            gitkeep_path.touch(exist_ok=True)
            logger.debug(f"Created .gitkeep: {gitkeep_path}")
        except OSError as e:
            logger.warning(f"Failed to create .gitkeep in {directory}: {e}")


def main() -> None:
    """Main entry point for directory creation."""
    # Get the project root (parent of scripts directory)
    script_path = Path(__file__).resolve()
    project_root = script_path.parent.parent
    
    logger.info(f"Project root: {project_root}")
    logger.info("Creating directory structure...")
    
    try:
        create_directories(project_root)
        create_gitkeep_files(project_root)
        logger.info("Directory structure created successfully.")
    except Exception as e:
        logger.error(f"Failed to create directory structure: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
