import numpy as np
from numpy.typing import NDArray
from typing import Optional
from array_utils import crop_array, bin_array, normalize_array
from config import Config

def preprocess_diffraction_data(diff3d: NDArray, 
                                normalization_strategy: Optional[str] = None,
                                crop_factor: int = 1,
                                bin_factor: int = 1) -> NDArray:
    """
    Preprocesses the diffraction data.

    Args:
        diff3d (NDArray): Input diffraction data (3D array).
        normalization_strategy (Optional[str]): Normalization method ('mean', 'max', or 'minmax'). Default is None (no normalization).
        crop_factor (int): Cropping factor. Default is 1 (no cropping).
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
    if crop_factor > 1:
        crop_shape = (diff3d.shape[0], diff3d.shape[1] // crop_factor, diff3d.shape[2] // crop_factor)
        diff3d = crop_array(diff3d, crop_shape)

    # Normalization
    if normalization_strategy is not None:
        diff3d = normalize_array(diff3d, method=normalization_strategy)

    return diff3d

def process_diffraction(diff3d: NDArray, config: Config, crop_factor: int, bin_factor: int) -> NDArray:
    """
    Main function to process the diffraction data according to the configuration.

    Args:
        diff3d (NDArray): Input diffraction data (3D array).
        config (Config): Configuration object containing processing parameters.
        crop_factor (int): Cropping factor.
        bin_factor (int): Binning factor.

    Returns:
        NDArray: Processed diffraction data.
    """
    return preprocess_diffraction_data(
        diff3d,
        normalization_strategy=config.diffraction_normalization_strategy,
        crop_factor=crop_factor,
        bin_factor=bin_factor
    )
