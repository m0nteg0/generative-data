## Generative-data.

This module is designed to prepare data that will later be used to train generative models, such as stable diffusion.  

Each input image is searched for a region containing a person using
the yolo detector. The found region is then reduced to square proportions,
preserving the person's face in the given region (if there is a face
in the image).

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

The model from the [project](https://github.com/akanametov/yolo-face) is used for face localization.  
To download model used in the project follow the code snippet below:
```bash
source download_models.sh
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
python crop_person.py --help
```
Example run:
```bash
cd scripts
python crop_person.py -i ../example/input -o ../example/output -s 512 --min-obj-size 256 --force
```
Arguments description:
```
-i: input images directory
-i: output images directory
-s: desired size of output images
--min-obj-size: minimal size of region that contains persons.
--force: forced overwrite of output directory
```

