import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union, Any

def _check_dimensions(arr: NDArray, dims: Tuple[int, ...]) -> None:
    if arr.ndim not in dims:
        raise ValueError(f"Input array must have {dims} dimensions.")

def _check_shapes_match(arr1: NDArray, arr2: NDArray) -> None:
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape.")

import numpy as np
from typing import Tuple, Union
from numpy.typing import NDArray

def crop_array(arr: NDArray, target_shape: Tuple[int, ...], shift: Union[Tuple[int, ...], None] = None) -> NDArray:
    """
    Crop the center of a N-dimensional array to the target shape, with optional shifting.
    Args:
        arr (NDArray): Input N-dimensional array.
        target_shape (Tuple[int, ...]): Desired output shape.
        shift (Union[Tuple[int, ...], None]): Tuple specifying the number of pixels to shift the crop region along each dimension (positive values shift down/right). If None, no shifting is applied.
    Returns:
        NDArray: Cropped array.
    Raises:
        ValueError: If the dimensions of shift do not match the dimensions of arr, or if the shifted crop region is outside the bounds of the input array.
    """
    if shift is None:
        shift = tuple(0 for _ in range(arr.ndim))
    elif len(shift) != arr.ndim:
        raise ValueError(f"The dimensions of shift ({len(shift)}) do not match the dimensions of arr ({arr.ndim}).")
    
    slices = []
    # Determine if we're dealing with a 3D array to apply cropping to the last two dimensions
    start_dim = arr.ndim - 2 if arr.ndim == 3 else 0
    
    for dim in range(start_dim, arr.ndim):
        start = (arr.shape[dim] - target_shape[dim]) // 2 + shift[dim]
        if start < 0 or start + target_shape[dim] > arr.shape[dim]:
            raise ValueError(f"Shifted crop region is outside the bounds of the input array along dimension {dim}.")
        slices.append(slice(start, start + target_shape[dim]))
    
    # Include any untouched dimensions from the beginning
    slices = [slice(None)] * start_dim + slices
    
    return arr[tuple(slices)]

import numpy as np

def bin_array(arr: np.ndarray, bin_factor: int) -> np.ndarray:
    """
    Bin a 2D or 3D array by averaging neighboring elements. If 3D, binning is applied to the last two dimensions.

    Args:
        arr (np.ndarray): Input 2D or 3D array.
        bin_factor (int): Binning factor.

    Returns:
        np.ndarray: Binned array.
    """
    if arr.ndim == 2:
        shape = (arr.shape[0] // bin_factor, arr.shape[1] // bin_factor)
        return arr.reshape(shape[0], bin_factor, shape[1], bin_factor).mean(axis=(1, 3))
    elif arr.ndim == 3:
        shape = (arr.shape[0], arr.shape[1] // bin_factor, arr.shape[2] // bin_factor)
        return arr.reshape(shape[0], shape[1], bin_factor, shape[2], bin_factor).mean(axis=(2, 4))
    else:
        raise ValueError("Input array must be 2D or 3D.")

def normalize_array(arr: NDArray, method: str = 'mean', axis: Union[int, Tuple[int, ...], None] = None) -> NDArray:
    """
    Normalizes an array using the specified method.
    """
    if method == 'mean':
        return arr / np.mean(arr, axis=axis, keepdims=True)
    elif method == 'max':
        return arr / np.max(arr, axis=axis, keepdims=True)
    elif method == 'minmax':
        min_val = np.min(arr, axis=axis, keepdims=True)
        max_val = np.max(arr, axis=axis, keepdims=True)
        return (arr - min_val) / (max_val - min_val)
    else:
        raise ValueError(f"Unsupported normalization method: {method}")

def pad_array(arr: NDArray, target_shape: Tuple[int, ...], mode: str = 'constant', constant_values: Any = 0) -> NDArray:
    """
    Pads an array to the target shape.
    """
    if len(target_shape) != arr.ndim:
        raise ValueError("Target shape must have the same number of dimensions as the input array.")
    
    if any(t < s for t, s in zip(target_shape, arr.shape)):
        raise ValueError("Target shape cannot be smaller than the input array shape in any dimension.")
    
    pad_width = [(0, t - s) for t, s in zip(target_shape, arr.shape)]
    
    return np.pad(arr, pad_width, mode=mode, constant_values=constant_values)

def apply_mask(arr: NDArray, mask: NDArray) -> NDArray:
    """
    Applies a mask to an array.
    """
    _check_shapes_match(arr, mask)
    return arr * mask
