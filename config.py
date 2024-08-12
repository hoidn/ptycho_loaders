from typing import NamedTuple, Tuple, Optional, Union

class Config(NamedTuple):
    # File paths (non-default fields)
    input_file: str
    output_file: str

    # Metadata (non-default fields)
    energy: float  # in eV
    detector_distance: float  # in meters
    pixel_size: float  # in meters

    # Probe processing parameters (with default values)
    probe_target_shape: Tuple[int, int] = (64, 64)
    probe_strategy: str = 'hybrid'
    probe_bin_factor: int = 2
    probe_shift_x: int = 0
    probe_shift_y: int = 0
    probe_scale: Optional[float] = None

    # Diffraction processing parameters (with default values)
    diffraction_normalization_strategy: Optional[str] = 'mean'

    @property
    def diffraction_crop_factor(self) -> int:
        return self.probe_bin_factor if self.probe_strategy != 'crop' else 1

    @property
    def diffraction_bin_factor(self) -> int:
        original_probe_shape = (512, 512)  # Assuming original probe shape is 512x512
        if self.probe_strategy == 'crop':
            return original_probe_shape[0] // self.probe_target_shape[0]
        else:
            return original_probe_shape[0] // (self.probe_target_shape[0] * self.diffraction_crop_factor)

    @property
    def diffraction_crop_shape(self) -> Tuple[int, int, int]:
        original_diff_shape = (512, 512, 512)  # Assuming original diffraction shape
        crop_size = original_diff_shape[0] // self.diffraction_crop_factor
        return (crop_size, crop_size, crop_size)

# Class constant (not a NamedTuple field)
Config.DEFAULT_PROBE_SCALE = 1.0

def load_config() -> Config:
    """
    Loads and returns the configuration.
    This function can be expanded to load from a file or environment variables.
    """
    return Config(
        # File paths
        input_file='input.cxi',
        output_file='output.npy',

        # Metadata
        energy=1000.0,  # 1 keV
        detector_distance=1.0,  # 1 meter
        pixel_size=55e-6,  # 55 micrometers

        # The rest of the fields will use their default values
    )

def update_config(config: Config, **kwargs) -> Config:
    """
    Updates the configuration with new values.

    Args:
        config (Config): The original configuration.
        **kwargs: Keyword arguments with new values to update.

    Returns:
        Config: An updated configuration.
    """
    return config._replace(**kwargs)

def validate_config(config: Config) -> None:
    """
    Validates the configuration.

    Args:
        config (Config): The configuration to validate.

    Raises:
        ValueError: If any configuration parameter is invalid.
    """
    if config.probe_bin_factor < 1:
        raise ValueError("probe_bin_factor must be 1 or greater")
    
    if config.diffraction_bin_factor < 1:
        raise ValueError("diffraction_bin_factor must be 1 or greater")
    
    if config.probe_strategy not in ['crop', 'bin', 'hybrid']:
        raise ValueError("probe_strategy must be 'crop', 'bin', or 'hybrid'")
    
    if config.diffraction_normalization_strategy not in [None, 'mean', 'max', 'minmax']:
        raise ValueError("diffraction_normalization_strategy must be None, 'mean', 'max', or 'minmax'")
    
    if config.energy <= 0:
        raise ValueError("energy must be positive")
    
    if config.detector_distance <= 0:
        raise ValueError("detector_distance must be positive")
    
    if config.pixel_size <= 0:
        raise ValueError("pixel_size must be positive")

def get_config() -> Config:
    """
    Loads, validates, and returns the configuration.

    Returns:
        Config: A validated configuration object.

    Raises:
        ValueError: If the configuration is invalid.
    """
    config = load_config()
    validate_config(config)
    return config
