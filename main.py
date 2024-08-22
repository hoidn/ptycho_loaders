"""
loader for David Shapiro's dataset
"""

import os
file_path = os.path.expanduser('~/Downloads/NS_231012075_ccdframes_0_0.cxi')

import matplotlib.pyplot as plt
import numpy as np

def visualize_probe(probe, title):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    im1 = ax1.imshow(np.abs(probe), cmap='viridis')
    ax1.set_title(f"{title} - Amplitude")
    ax1.set_xlabel("Pixel X")
    ax1.set_ylabel("Pixel Y")
    fig.colorbar(im1, ax=ax1, label="Amplitude")
    
    im2 = ax2.imshow(np.angle(probe), cmap='hsv')
    ax2.set_title(f"{title} - Phase")
    ax2.set_xlabel("Pixel X")
    ax2.set_ylabel("Pixel Y")
    fig.colorbar(im2, ax=ax2, label="Phase (radians)")
    
    plt.tight_layout()
    plt.show()

def plot_frame(frame, frame_idx):
    plt.figure(figsize=(8, 8))
    plt.imshow(frame, cmap='viridis')
    plt.colorbar(label='Intensity')
    plt.title(f"Frame (Index: {frame_idx})")
    plt.xlabel("Pixel X")
    plt.ylabel("Pixel Y")
    plt.show()

from config import Config, update_config, get_config
from pipeline import create_processing_pipeline
from file_io import save_preprocessed_data, load_preprocessed_data

# Start with the default configuration
config = get_config()

# Update the configuration with our specific parameters
config = update_config(config, 
                       input_file=file_path,
                       output_file='als128.npy',
                       # output_file='als512.npy',
                       probe_bin_factor=2,
                       probe_target_shape=(128, 128))

# Create and run the pipeline
pipeline = create_processing_pipeline(config)
result = pipeline(config.input_file)

# Save the preprocessed data
save_preprocessed_data(config.output_file, **result)

# Load the saved data to extract specific arrays
loaded_data = load_preprocessed_data('als128.npy.npz')
# loaded_data = load_preprocessed_data('als512.npy.npz')

# Extract the required arrays
d3d = loaded_data['diffraction']
probe = loaded_data['probeGuess']
scan_index = loaded_data['scan_index']
objectGuess = loaded_data['objectGuess']
xcoords = loaded_data['xcoords']
ycoords = loaded_data['ycoords']
