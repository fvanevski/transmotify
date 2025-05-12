# utils/file_manager.py
"""
Utility functions for file and directory management, including temporary files,
directory creation, cleanup, and archive creation.
"""

import os
import shutil
import tempfile
import zipfile
import traceback
from pathlib import Path
from typing import List, Optional, TextIO, Union, Dict, Any
from config.config import (
    Config,
)  # Assuming config.py is in the same directory as this module

# Importing Config from config.py, assuming it contains necessary configuration settings

# Assuming core.logging is established from Phase 1
try:
    from core.logging import log_info, log_warning, log_error
except ImportError:
    # Fallback basic print logging if core.logging is unavailable
    def log_error(message: str):
        print(f"ERROR (logging unavailable): {message}")

    def log_warning(message: str):
        print(f"WARNING (logging unavailable): {message}")

    def log_info(message: str):
        print(f"INFO (logging unavailable): {message}")


def __init__(self, config: Config):
    self.config = config


# Moved from core/utils.py
def create_directory(path: Union[str, Path]) -> bool:
    """Creates a directory if it doesn't exist. Returns True on success."""
    dir_path = Path(path)
    try:
        dir_path.mkdir(parents=True, exist_ok=True)
        log_info(f"Ensured directory exists: {dir_path}")
        return True
    except OSError as e:
        log_warning(f"Could not create directory {dir_path}: {e}")
        return False
    except Exception as e:
        log_error(f"Unexpected error creating directory {dir_path}: {e}")
        return False


# Refactored from core/utils.py:get_temp_file
# Renamed for clarity
def get_temp_file_path(temp_dir_base_path: Path, suffix: str = "") -> Optional[Path]:
    """Creates a temporary file and returns its Path object, or None on failure."""
    try:
        # Note: This creates a file in the system's default temp location,
        # NOT necessarily within the project's configured temp_dir.
        # Consider adding a version that takes a base directory if needed.
        config = Config()  # Instantiate Config
        config_temp_dir = Path(config.get("temp_dir", "./temp"))
        log_info(f"Using temporary directory: {config_temp_dir}")
        # Ensure the temp directory exists
        if not config_temp_dir.exists():
            log_warning(
                f"Temporary directory does not exist: {config_temp_dir}. Creating it."
            )
            create_directory(config_temp_dir)
        temp_file = tempfile.mkstemp(dir=config_temp_dir, suffix=suffix)
        temp_file_path: Path = Path(temp_file.name)
        temp_file.close()
        log_info(f"Created temporary file: {temp_file_path}")
        return temp_file_path
    except Exception as e:
        log_error(f"Failed to create temporary file: {e}")
        return None


# Refactored from core/file_management.py:FileManager.save_file
def save_text_file(content: str, file_path: Union[str, Path]) -> bool:
    """Saves string content to a file. Ensures parent directory exists."""
    path_obj = Path(file_path)
    log_info(f"Attempting to save text file to: {path_obj}")
    if not create_directory(path_obj.parent):
        log_error(f"Failed to create parent directory for {path_obj}. Aborting save.")
        return False
    try:
        with open(path_obj, "w", encoding="utf-8") as f:
            f.write(content)
        log_info(f"File saved successfully: {path_obj}")
        return True
    except IOError as e:
        log_error(f"Failed to save file {path_obj}: {e}")
        return False
    except Exception as e:
        log_error(f"Unexpected error saving file {path_obj}: {e}")
        return False


# Refactored from core/file_management.py:FileManager.cleanup_temp_files
# Made more generic: cleans ANY directory.
def cleanup_directory(dir_path: Union[str, Path], recreate: bool = True) -> bool:
    """Removes and optionally recreates a directory."""
    path_obj = Path(dir_path)
    log_info(f"Attempting cleanup for directory: {path_obj} (recreate={recreate})")
    try:
        if path_obj.exists():
            shutil.rmtree(path_obj)
            log_info(f"Removed directory: {path_obj}")
        else:
            log_info(f"Directory does not exist, no removal needed: {path_obj}")

        if recreate:
            return create_directory(path_obj)
        return True
    except OSError as e:
        log_error(f"Failed cleanup for directory {path_obj}: {e}")
        return False
    except Exception as e:
        log_error(f"Unexpected error during cleanup for {path_obj}: {e}")
        return False


# Refactored from core/pipeline.py:create_final_zip
def create_zip_archive(
    zip_path: Union[str, Path],
    files_to_add: Dict[str, Union[str, Path]],  # Arcname -> Source Path
    log_prefix: str = "",
) -> Optional[Path]:
    """Creates a ZIP archive from a dictionary of files."""
    zip_path_obj = Path(zip_path)
    log_info(f"{log_prefix}Attempting to create ZIP archive: {zip_path_obj}")

    if not files_to_add:
        log_warning(
            f"{log_prefix}No files provided to add to the zip archive. Skipping zip creation."
        )
        return None

    # Ensure output directory exists
    if not create_directory(zip_path_obj.parent):
        log_error(
            f"{log_prefix}Failed to create output directory {zip_path_obj.parent}. Aborting zip."
        )
        return None

    # Use a temporary zip file during creation
    temp_zip_path = zip_path_obj.with_suffix(f".{os.getpid()}.temp.zip")
    files_added_count = 0
    files_skipped_count = 0

    try:
        with zipfile.ZipFile(str(temp_zip_path), "w", zipfile.ZIP_DEFLATED) as zf:
            for arc_name, local_path_src in files_to_add.items():
                try:
                    local_path = Path(local_path_src)  # Ensure it's a Path object
                except TypeError:
                    log_warning(
                        f"{log_prefix}Invalid path type for archive name '{arc_name}', skipping: {local_path_src}"
                    )
                    files_skipped_count += 1
                    continue

                if local_path.is_file():
                    try:
                        zf.write(local_path, arcname=arc_name)
                        files_added_count += 1
                        log_info(
                            f"{log_prefix}Added to zip: '{local_path}' as '{arc_name}'"
                        )
                    except Exception as e:
                        log_warning(
                            f"{log_prefix}Failed to add file {local_path} to zip as {arc_name}: {e}"
                        )
                        files_skipped_count += 1
                else:
                    log_warning(
                        f"{log_prefix}File not found or is not a file, skipping: {local_path}"
                    )
                    files_skipped_count += 1

        if files_added_count == 0:
            log_error(
                f"{log_prefix}No files were successfully added to the zip. Aborting zip creation."
            )
            if temp_zip_path.exists():
                temp_zip_path.unlink(missing_ok=True)  # Use missing_ok=True
            return None

        # Move temporary zip to final location
        shutil.move(str(temp_zip_path), str(zip_path_obj))
        log_info(f"{log_prefix}Successfully created final ZIP: {zip_path_obj}")
        log_info(
            f"{log_prefix}Files added: {files_added_count}, Files skipped: {files_skipped_count}"
        )
        return zip_path_obj

    except Exception as e:
        log_error(
            f"{log_prefix}Failed to create ZIP file {zip_path_obj}: {e}\n{traceback.format_exc()}"
        )
        if temp_zip_path.exists():
            temp_zip_path.unlink(missing_ok=True)
            log_info(f"{log_prefix}Cleaned up temporary zip file.")
        return None


# Extracted from core/pipeline.py:_prepare_audio_input
def copy_local_file(
    source_path: Union[str, Path],
    destination_dir: Union[str, Path],
    destination_filename: Optional[str] = None,
    log_prefix: str = "",
) -> Optional[Path]:
    """Copies a local file to a destination directory."""
    src_path = Path(source_path)
    dest_dir = Path(destination_dir)

    if not src_path.is_file():
        log_error(f"{log_prefix}Source file not found or is not a file: {src_path}")
        return None

    if not create_directory(dest_dir):
        log_error(
            f"{log_prefix}Failed to ensure destination directory exists: {dest_dir}"
        )
        return None

    dest_filename = destination_filename or src_path.name
    dest_path = dest_dir / dest_filename

    try:
        log_info(f"{log_prefix}Copying local file {src_path} to {dest_path}")
        shutil.copy(str(src_path), str(dest_path))
        log_info(f"{log_prefix}File copied successfully.")
        return dest_path
    except Exception as e:
        log_error(
            f"{log_prefix}Failed to copy file from {src_path} to {dest_path}: {e}"
        )
        log_error(traceback.format_exc())
        return None
