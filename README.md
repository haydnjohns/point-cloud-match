# Point Cloud Match

This project demonstrates affine registration of 2D and 3D point clouds using the Coherent Point Drift (CPD) algorithm. It includes scripts for generating synthetic point clouds with known transformations and performing registration to estimate those transformations.

## Project Structure

```
point-cloud-match
├── .gitignore  
├── README.md  
├── assets  
│   ├── 2D_example_output.png  
│   └── 3D_example_output.png  
├── requirements.txt  
└── src  
    ├── 2D_cpd.py  
    └── 3D_cpd.py  
```

- `src/2D_cpd.py`: Generates 2D synthetic point clouds, applies known affine transformations, and registers them using CPD.  
- `src/3D_cpd.py`: Similar to `2D_cpd.py` but for 3D point clouds with rotation, scaling, translation along specific axes.  
- `assets/`: Contains example output images demonstrating registration results.

## Installation

Create a Python virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the scripts from the `src` directory:

```bash
python 2D_cpd.py
python 3D_cpd.py
```

Each script will generate point clouds, perform affine registration, print transformation estimates and errors, and display before/after plots.

## Dependencies

- numpy  
- matplotlib  
- pycpd  

These are listed in `requirements.txt`.
