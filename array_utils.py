import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Union, Any

def _check_dimensions(arr: NDArray, dims: Tuple[int, ...]) -> None:
    if arr.ndim not in dims:
        raise ValueError(f"Input array must have {dims} dimensions.")

def _check_shapes_match(arr1: NDArray, arr2: NDArray) -> None:
    if arr1.shape != arr2.shape:
        raise ValueError("Input arrays must have the same shape.")

def crop_array(arr: NDArray, target_shape: Tuple[int, ...], shift: Union[Tuple[int, ...], None] = None) -> NDArray:
    """
    Crops the center of an array to the target shape. Works with arrays of any dimension.
    """
    if len(target_shape) != arr.ndim:
        raise ValueError("Target shape must have the same number of dimensions as the input array.")
    
    if any(t > s for t, s in zip(target_shape, arr.shape)):
        raise ValueError("Target shape cannot be larger than the input array shape in any dimension.")
    
    shift = shift or (0,) * arr.ndim
    
    if len(shift) != arr.ndim:
        raise ValueError("Shift must have the same number of dimensions as the input array.")
    
    starts = [max(0, (s - t) // 2 + sh) for s, t, sh in zip(arr.shape, target_shape, shift)]
    ends = [min(s, start + t) for s, start, t in zip(arr.shape, starts, target_shape)]
    
    slices = tuple(slice(s, e) for s, e in zip(starts, ends))
    return arr[slices]

def bin_array(arr: NDArray, bin_factor: int) -> NDArray:
    """
    Bins an array by averaging neighboring elements. Works with arrays of any dimension.
    """
    if bin_factor < 1:
        raise ValueError("Bin factor must be a positive integer.")
    
    if bin_factor == 1:
        return arr
    
    if any(s % bin_factor != 0 for s in arr.shape):
        raise ValueError("All array dimensions must be divisible by the bin factor.")
    
    new_shape = tuple(s // bin_factor for s in arr.shape) + (bin_factor,) * arr.ndim
    return arr.reshape(new_shape).mean(axis=tuple(range(arr.ndim, 2*arr.ndim)))

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
