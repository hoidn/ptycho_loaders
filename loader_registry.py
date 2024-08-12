import importlib
from typing import Dict, Callable, Any
import os

def get_loader(file_path: str) -> Dict[str, Callable]:
    """
    Returns a dictionary of loader functions based on the file extension.

    Args:
        file_path (str): Path to the file to be loaded.

    Returns:
        Dict[str, Callable]: A dictionary containing loader functions.

    Raises:
        ValueError: If the file extension is not supported.
    """
    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    if file_extension == '.cxi':
        module_name = 'loaders.cxi_loader'
    elif file_extension == '.npy':
        module_name = 'loaders.npy_loader'
    else:
        raise ValueError(f"Unsupported file extension: {file_extension}")

    try:
        loader_module = importlib.import_module(module_name)
        return {
            'load_probe': loader_module.load_probe,
            'load_diffraction_data': loader_module.load_diffraction_data,
            'load_scan_positions': loader_module.load_scan_positions,
            'load_scan_index': loader_module.load_scan_index,
            'load_object_guess': loader_module.load_object_guess
        }
    except ImportError as e:
        raise ImportError(f"Error importing loader module {module_name}: {str(e)}")
    except AttributeError as e:
        raise AttributeError(f"Error accessing loader functions in module {module_name}: {str(e)}")
