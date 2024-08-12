import logging
from typing import Callable, Dict, Any
from config import Config
from loader_registry import get_loader
from probe_processing import process_probe
from diffraction_processing import process_diffraction
from file_io import save_preprocessed_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_processing_pipeline(config: Config) -> Callable[[str], Dict[str, Any]]:
    def pipeline(input_file: str) -> Dict[str, Any]:
        logger.info(f"Starting processing pipeline for {input_file}")
        
        loader = get_loader(input_file)
        required_functions = ['load_probe', 'load_diffraction_data', 'load_scan_positions', 'load_scan_index', 'load_object_guess']
        
        for func in required_functions:
            if func not in loader:
                raise ValueError(f"Loader is missing required function: {func}")

        # Load data
        logger.info("Loading data...")
        probe_data = loader['load_probe'](input_file)
        diff3d = loader['load_diffraction_data'](input_file)
        xcoords, ycoords = loader['load_scan_positions'](input_file)
        scan_index = loader['load_scan_index'](input_file)
        object_guess = loader['load_object_guess'](input_file)

        # Calculate diffraction crop_factor and bin_factor based on probe parameters
        if config.probe_strategy == 'crop':
            bin_factor = config.initial_shape // config.probe_target_shape
            crop_factor = 1
        else:
            crop_factor = config.probe_bin_factor
            bin_factor = config.initial_shape // (config.probe_target_shape[0] * crop_factor)

        # Process data
        logger.info("Processing probe data...")
        processed_probe = process_probe(probe_data, config)
        logger.info("Processing diffraction data...")
        processed_diff3d = process_diffraction(diff3d, config, crop_factor, bin_factor)

        logger.info("Pipeline processing complete")
        return {
            'probe': processed_probe,
            'diffraction': processed_diff3d,
            'scan_index': scan_index,
            'object_guess': object_guess,
            'xcoords': xcoords,
            'ycoords': ycoords,
        }

    return pipeline
