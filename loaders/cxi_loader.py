import h5py
import numpy as np
from typing import Tuple, Any

def get_val(file_path: str, key: str) -> Any:
    """
    Helper function to get a value from an H5 file.

    Args:
        file_path (str): Path to the H5 file.
        key (str): Key to retrieve the value.

    Returns:
        Any: The value associated with the key.
    """
    with h5py.File(file_path, 'r') as f:
        try:
            return f[key][:]
        except ValueError:
            return f[key][()]

def load_probe(file_path: str) -> np.ndarray:
    """
    Loads probe data from a CXI file.

    Args:
        file_path (str): Path to the CXI file.

    Returns:
        np.ndarray: The probe data.

    Raises:
        ValueError: If the probe dataset is not found in the file.
        IOError: If there's an error reading the CXI file.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            probe_dataset = f.get('entry_1/instrument_1/detector_1/probe') or f.get('entry_1/instrument_1/source_1/probe')
            if probe_dataset is None:
                raise ValueError("Probe dataset not found in the .cxi file.")
            return probe_dataset[:]
    except IOError as e:
        raise IOError(f"Error reading .cxi file: {str(e)}")

def load_diffraction_data(file_path: str) -> np.ndarray:
    """
    Loads diffraction data from a CXI file.

    Args:
        file_path (str): Path to the CXI file.

    Returns:
        np.ndarray: The diffraction data.
    """
    with h5py.File(file_path, 'r') as f:
        diff3d = f['entry_1/instrument_1/detector_1/data'][:]
    return diff3d

def load_scan_positions(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Loads scan positions from a CXI file.

    Args:
        file_path (str): Path to the CXI file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The x and y coordinates of scan positions.
    """
    with h5py.File(file_path, 'r') as f:
        dx = get_val(file_path, 'entry_1/instrument_1/detector_1/x_pixel_size')
        dy = get_val(file_path, 'entry_1/instrument_1/detector_1/y_pixel_size')
        dx = dy = 1
        
        xcoords = f['entry_1/sample_1/geometry_1/translation'][:, 0] / dx
        ycoords = f['entry_1/sample_1/geometry_1/translation'][:, 1] / dy
    return xcoords, ycoords

def load_scan_index(file_path: str) -> np.ndarray:
    """
    Loads scan index from a CXI file.

    Args:
        file_path (str): Path to the CXI file.

    Returns:
        np.ndarray: The scan index.
    """
    with h5py.File(file_path, 'r') as f:
        scan_index = np.zeros(f['entry_1/instrument_1/detector_1/data'].shape[0])
    return scan_index

def load_object_guess(file_path: str) -> np.ndarray:
    """
    Loads object guess from a CXI file.

    Args:
        file_path (str): Path to the CXI file.

    Returns:
        np.ndarray: The object guess data.
    """
    return get_val(file_path, 'entry_1/image_1/data')
