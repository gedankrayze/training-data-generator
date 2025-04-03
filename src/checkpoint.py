"""
Checkpoint management for error recovery.
"""

import hashlib
import logging
import pickle
from pathlib import Path
from typing import Any, Optional

# Configure logging
logger = logging.getLogger("checkpoint")


class CheckpointManager:
    """
    Manages checkpoints for long-running processes to enable error recovery.
    """

    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """
        Initialize the checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoint files
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True, parents=True)

    def _get_checkpoint_path(self, key: str) -> Path:
        """Get the path for a checkpoint file based on a key."""
        # Create a hash of the key for the filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.checkpoint_dir / f"checkpoint_{key_hash}.pkl"

    def save_checkpoint(self, key: str, data: Any) -> bool:
        """
        Save data to a checkpoint file.
        
        Args:
            key: Unique identifier for the checkpoint
            data: Data to save in the checkpoint
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            checkpoint_path = self._get_checkpoint_path(key)
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f)
            logger.info(f"Saved checkpoint: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to save checkpoint {key}: {e}")
            return False

    def load_checkpoint(self, key: str) -> Optional[Any]:
        """
        Load data from a checkpoint file.
        
        Args:
            key: Unique identifier for the checkpoint
            
        Returns:
            The loaded data or None if not found/error
        """
        try:
            checkpoint_path = self._get_checkpoint_path(key)
            if not checkpoint_path.exists():
                return None

            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            logger.info(f"Loaded checkpoint: {key}")
            return data
        except Exception as e:
            logger.error(f"Failed to load checkpoint {key}: {e}")
            return None

    def checkpoint_exists(self, key: str) -> bool:
        """Check if a checkpoint exists."""
        return self._get_checkpoint_path(key).exists()

    def delete_checkpoint(self, key: str) -> bool:
        """Delete a checkpoint file."""
        try:
            checkpoint_path = self._get_checkpoint_path(key)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"Deleted checkpoint: {key}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete checkpoint {key}: {e}")
            return False
