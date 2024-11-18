## Generative-data.

This module is designed to prepare data that will later be used to train generative models, such as stable diffusion.

## Getting Started

Install using conda environment (recommended):

```bash
conda create -n generative-data -y python=3.10
conda activate generative-data

pip install -e .
```

Install using venv environment (python=3.10 is required):

```bash
python -m venv .venv
source .venv/bin/actiavte

pip install -e .
```
### Running from commandline

Running a script to crop images with people:
```bash
# Go to scripts directory:
cd scripts
# Run script:
python crop_person.py -i <input_images_dir> -o <output_dir>
# Run script with specified number of random images from each subdirectory
python crop_person.py -i <input_images_dir> -o <output_dir> --limit-per-subdir <number>
# To see all possible options, run the command:
python crop_person.py -h
```