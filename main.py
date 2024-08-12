import argparse
import logging
from config import Config, load_config
from pipeline import create_processing_pipeline
from file_io import save_preprocessed_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Preprocess CXI data for ptychography.")
    parser.add_argument('input_file', help="Path to the input CXI file")
    parser.add_argument('output_file', help="Path to save the output NPY file")
    parser.add_argument('--probe-bin-factor', type=int, help="Override probe bin factor")
    args = parser.parse_args()

    try:
        config = load_config()
        config = config._replace(input_file=args.input_file, output_file=args.output_file)
        
        if args.probe_bin_factor is not None:
            config = config._replace(probe_bin_factor=args.probe_bin_factor)

        pipeline = create_processing_pipeline(config)
        result = pipeline(config.input_file)

        save_preprocessed_data(config.output_file, **result)

        logger.info(f"Preprocessing complete. Output saved to {config.output_file}")
    except Exception as e:
        logger.error(f"An error occurred during processing: {str(e)}")
        raise

if __name__ == "__main__":
    main()
