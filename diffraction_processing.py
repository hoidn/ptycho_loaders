import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from array_utils import crop_array, bin_array, normalize_array
from config import Config

def preprocess_diffraction_data(diff3d: NDArray, 
                                normalization_strategy: Optional[str] = None,
                                crop_shape: Optional[Tuple[int, int, int]] = None, 
                                bin_factor: int = 1) -> NDArray:
    """
    Preprocesses the diffraction data.

    Args:
        diff3d (NDArray): Input diffraction data (3D array).
        normalization_strategy (Optional[str]): Normalization method ('mean', 'max', or 'minmax'). Default is None (no normalization).
        crop_shape (Optional[Tuple[int, int, int]]): Shape to crop the data to. Default is None (no cropping).
        bin_factor (int): Binning factor. Default is 1 (no binning).

    Returns:
        NDArray: Preprocessed diffraction data.
    """
    if diff3d.ndim != 3:
        raise ValueError("Input diffraction data must be a 3D array.")

    # Binning
    if bin_factor > 1:
        diff3d = bin_array(diff3d, bin_factor)

    # Cropping
    if crop_shape is not None:
        diff3d = crop_array(diff3d, crop_shape)

    # Normalization
    if normalization_strategy is not None:
        diff3d = normalize_array(diff3d, method=normalization_strategy)

    return diff3d

def process_diffraction(diff3d: NDArray, config: Config) -> NDArray:
    """
    Main function to process the diffraction data according to the configuration.

    Args:
        diff3d (NDArray): Input diffraction data (3D array).
        config (Config): Configuration object containing processing parameters.

    Returns:
        NDArray: Processed diffraction data.
    """
    return preprocess_diffraction_data(
        diff3d,
        normalization_strategy=config.diffraction_normalization_strategy,
        crop_shape=config.diffraction_crop_shape,
        bin_factor=config.diffraction_bin_factor
    )
