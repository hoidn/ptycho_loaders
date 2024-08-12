import numpy as np
from typing import Dict, Any

def save_preprocessed_data(file_path: str, 
                           diffraction: np.ndarray, 
                           probe: np.ndarray,
                           scan_index: np.ndarray, 
                           object_guess: np.ndarray,  # Changed from objectGuess to object_guess
                           xcoords: np.ndarray, 
                           ycoords: np.ndarray) -> None:
    """
    Saves preprocessed data to a file.

    Args:
        file_path (str): Path to save the preprocessed data.
        diffraction (np.ndarray): Preprocessed diffraction data.
        probe (np.ndarray): Preprocessed probe data.
        scan_index (np.ndarray): Scan index data.
        object_guess (np.ndarray): Object guess data.
        xcoords (np.ndarray): X coordinates.
        ycoords (np.ndarray): Y coordinates.

    Raises:
        IOError: If there's an error saving the file.
    """
    try:
        np.savez(file_path, 
                 diffraction=diffraction, 
                 probeGuess=probe,
                 scan_index=scan_index,
                 objectGuess=object_guess,  # Keep the original name in the saved file
                 xcoords_start=xcoords, 
                 ycoords_start=ycoords,
                 xcoords=xcoords, 
                 ycoords=ycoords)
        print(f"Preprocessed data saved to {file_path}")
    except IOError as e:
        raise IOError(f"Error saving preprocessed data to {file_path}: {str(e)}")

def load_preprocessed_data(file_path: str) -> Dict[str, np.ndarray]:
    """
    Loads preprocessed data from a file.

    Args:
        file_path (str): Path to the preprocessed data file.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the loaded data.

    Raises:
        IOError: If there's an error loading the file.
    """
    try:
        with np.load(file_path) as data:
            return {key: data[key] for key in data.files}
    except IOError as e:
        raise IOError(f"Error loading preprocessed data from {file_path}: {str(e)}")
