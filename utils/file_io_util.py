import os
import json
import pickle
import logging
from typing import Any, Dict, Union

# Configure logger for this module
logger = logging.getLogger("File IO Util")

def save_json(data: Union[Dict, list], file_path: str, indent: int = 4):
    """
    Save data to a JSON file.
    
    Args:
        data: The dictionary or list to save.
        file_path: The target file path.
        indent: Indentation level for pretty printing.
    """
    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok = True)
        
        with open(file_path, 'w', encoding = 'utf-8') as f:
            json.dump(data, f, indent = indent, ensure_ascii = False)
        logger.info(f"Successfully saved JSON to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {file_path}: {e}")

def load_json(file_path: str) -> Union[Dict, list]:
    """
    Load data from a JSON file.
    
    Args:
        file_path: The path to the JSON file.
        
    Returns:
        The loaded data (dict or list).
    """
    try:
        with open(file_path, 'r', encoding = 'utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Failed to load JSON from {file_path}: {e}")
        return {}

def save_pickle(obj: Any, file_path: str):
    """
    Save a Python object to a pickle file.
    """
    try:
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok = True)
        with open(file_path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"Successfully saved object to {file_path}")
    except Exception as e:
        logger.error(f"Failed to save pickle to {file_path}: {e}")
