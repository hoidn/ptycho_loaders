import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from array_utils import crop_array, bin_array, normalize_array, apply_mask
from config import Config

def preprocess_probe(probe: NDArray, 
                     target_shape: Tuple[int, int], 
                     strategy: str, 
                     shift: Tuple[int, int] = (0, 0), 
                     bin_factor: int = 1) -> NDArray:
    """
    Preprocesses the probe data according to the specified strategy.

    Args:
        probe (NDArray): Input probe data (2D array).
        target_shape (Tuple[int, int]): Desired output shape.
        strategy (str): Preprocessing strategy ('crop', 'bin', or 'hybrid').
        shift (Tuple[int, int]): Shift from center for cropping. Default is (0, 0).
        bin_factor (int): Binning factor. Default is 1 (no binning).

    Returns:
        NDArray: Preprocessed probe data.

    Raises:
        ValueError: If an invalid strategy is specified.
    """
    if probe.ndim != 2:
        raise ValueError("Input probe must be a 2D array.")

    if strategy == 'crop':
        return crop_array(probe, target_shape, shift)
    elif strategy == 'bin':
        binned = bin_array(probe, bin_factor)
        return crop_array(binned, target_shape, shift)
    elif strategy == 'hybrid':
        if bin_factor > 1:
            probe = bin_array(probe, bin_factor)
        return crop_array(probe, target_shape, shift)
    else:
        raise ValueError(f"Invalid preprocessing strategy: {strategy}")

def normalize_probe(probe: NDArray, mask: NDArray, probe_scale: Optional[float] = None) -> NDArray:
    """
    Normalizes the probe data.

    Args:
        probe (NDArray): Input probe data (2D array).
        mask (NDArray): Mask to apply before normalization (2D array).
        probe_scale (Optional[float]): Scaling factor for normalization. If None, uses the default from Config.

    Returns:
        NDArray: Normalized probe data.
    """
    if probe_scale is None:
        probe_scale = Config.DEFAULT_PROBE_SCALE

    masked_probe = apply_mask(probe, mask)
    norm_factor = probe_scale * np.mean(np.abs(masked_probe))
    return probe / norm_factor

def process_probe(probe: NDArray, config: Config) -> NDArray:
    """
    Main function to process the probe data according to the configuration.

    Args:
        probe (NDArray): Input probe data (2D array).
        config (Config): Configuration object containing processing parameters.

    Returns:
        NDArray: Processed probe data.
    """
    preprocessed_probe = preprocess_probe(
        probe, 
        config.probe_target_shape, 
        config.probe_strategy,
        (config.probe_shift_y, config.probe_shift_x),
        config.probe_bin_factor
    )

    mask = get_probe_mask(preprocessed_probe.shape[0])  # Assuming square probe
    normalized_probe = normalize_probe(preprocessed_probe, mask, config.probe_scale)

    return normalized_probe

def get_probe_mask(N: int) -> NDArray:
    """
    Generate a circular mask for the probe.

    Args:
        N (int): Size of the square mask.

    Returns:
        NDArray: Circular mask (2D array).
    """
    y, x = np.ogrid[-N//2:N//2, -N//2:N//2]
    mask = x*x + y*y <= (N//2)*(N//2)
    return mask.astype(float)
