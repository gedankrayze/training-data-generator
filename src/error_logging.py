"""
Error logging functionality for tracking and analyzing generation errors.
"""

import datetime
import json
import logging
import os
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger("error_logging")


class ErrorLogger:
    """Class for logging errors to a JSON file."""

    def __init__(self, error_file_path: str = "errors.json"):
        """
        Initialize the error logger.
        
        Args:
            error_file_path: Path to the error log file
        """
        self.error_file_path = error_file_path
        self.errors = self._load_existing_errors()

    def _load_existing_errors(self) -> List[Dict[str, Any]]:
        """Load existing errors from the error file if it exists."""
        if os.path.exists(self.error_file_path):
            try:
                with open(self.error_file_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.warning(f"Error file {self.error_file_path} is not valid JSON. Creating new file.")
                return []
        return []

    def log_error(self,
                  error_type: str,
                  error_message: str,
                  chunk_info: Optional[Dict[str, Any]] = None,
                  response_text: Optional[str] = None,
                  additional_info: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error to the error file.
        
        Args:
            error_type: Type of error (e.g., 'api_error', 'parsing_error', 'validation_error')
            error_message: Error message
            chunk_info: Information about the document chunk being processed
            response_text: Text of the LLM response that caused the error
            additional_info: Any additional information to include
        """
        # Create error record
        error_record = {
            "timestamp": datetime.datetime.now().isoformat(),
            "error_type": error_type,
            "error_message": error_message,
        }

        # Add chunk info if provided
        if chunk_info:
            # Only include relevant fields to avoid giant error logs
            filtered_chunk_info = {
                "file_path": chunk_info.get("file_path", ""),
                "file_name": chunk_info.get("file_name", ""),
                "chunk_id": chunk_info.get("chunk_id", ""),
                "chunk_total": chunk_info.get("chunk_total", ""),
                # Include a preview of the content, not the full content
                "content_preview": chunk_info.get("content", "")[:200] + "..." if chunk_info.get("content") else ""
            }
            error_record["chunk_info"] = filtered_chunk_info

        # Add response text if provided
        if response_text:
            error_record["response_text"] = response_text

        # Add additional info if provided
        if additional_info:
            error_record["additional_info"] = additional_info

        # Add error to list
        self.errors.append(error_record)

        # Write errors to file
        self._write_errors_to_file()

        # Log to console as well
        logger.error(f"Error logged: {error_type} - {error_message}")

    def _write_errors_to_file(self) -> None:
        """Write errors to the error file."""
        try:
            with open(self.error_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.errors, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to write errors to file: {e}")

    def get_errors(self) -> List[Dict[str, Any]]:
        """Get all errors."""
        return self.errors

    def get_error_summary(self) -> Dict[str, int]:
        """Get a summary of errors by error type."""
        summary = {}
        for error in self.errors:
            error_type = error["error_type"]
            summary[error_type] = summary.get(error_type, 0) + 1
        return summary

    def clear_errors(self) -> None:
        """Clear all errors."""
        self.errors = []
        self._write_errors_to_file()
