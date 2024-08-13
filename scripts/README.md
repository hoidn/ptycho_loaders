# Ptychography Data Loading and Reconstruction Script

This script is designed to load ptychography data from an HDF5 file, perform optional reconstruction, and save the results to a NumPy NPZ file. It is intended to be used as part of a larger workflow, generating training data for a downstream model training step.

## Usage

The script can be run from the command line with the following syntax:

```
python ptycho_script.py EXPERIMENT RUN [OPTIONS]
```

Required Arguments:
- `EXPERIMENT`: The experiment name (e.g., xppd00120)
- `RUN`: The run number (e.g., 1084)

Optional Arguments:
- `-m, --mode`: Script mode, either `load_only` (default) or `load_recon`
- `-w, --width`: Width parameter, either 64 (default) or 128
- `-o, --output_prefix`: Output file prefix (default: "ptycho_output")
- `-c, --config`: Path to YAML configuration file for metadata

Metadata Arguments (used if no config file provided):
- `--low_thresh`: Low threshold (default: 40000)
- `--high_thresh`: High threshold (default: 50000)
- `--center_x`: Center X (default: 150)
- `--center_y`: Center Y (default: 150)  
- `--im_thresh`: Image threshold (default: 20)
- `--z`: Z value (default: 4.147)
- `--lambda0`: Lambda0 value (default: 1239.8/8889*1e-9)
- `--pD`: pD value (default: 75e-6)
- `--angle`: Angle (default: 180)

## Modes

The script can be run in two modes:

1. `load_only`: Loads the data from the HDF5 file and saves the following arrays to an NPZ file:
   - `diffraction`
   - `xcoords`
   - `ycoords` 
   - `xcoords_start`
   - `ycoords_start`

2. `load_recon`: Loads the data, performs ptychographic reconstruction using the specified recipe, and saves the following arrays to an NPZ file:
   - `diffraction`
   - `probeGuess`
   - `objectGuess`
   - `xcoords`
   - `ycoords`
   - `xcoords_start` 
   - `ycoords_start`

## YAML Configuration

Metadata parameters can be provided via a YAML configuration file. If a config file is not specified, the script will use the default values or the values provided via command line arguments.

Example YAML config:

```yaml
low_thresh: 45000
high_thresh: 55000
center_x: 160
center_y: 160
im_thresh: 25
z: 4.2
lambda0: 1.4e-10
pD: 80e-6  
angle: 90
```

## Example Usage

Load data only with default parameters:
```
python ptycho_script.py xppd00120 1084
```

Load data and perform reconstruction with width 128 and custom output prefix:
```
python ptycho_script.py xppd00120 1084 -m load_recon -w 128 -o my_output 
```

Load data with metadata from YAML file:
```
python ptycho_script.py xppd00120 1084 -c metadata.yaml
```

Load data with metadata from command line arguments:
```
python ptycho_script.py xppd00120 1084 --low_thresh 45000 --high_thresh 55000 --center_x 160
```
