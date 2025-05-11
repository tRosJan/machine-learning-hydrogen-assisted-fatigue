# Hydrogen-assisted-fatigue-ANN

## Project Structure

The repository has the following structure:

- `/dataset/` - contains the datasets.
- `/developed model/` - includes the developed model and usage notebook.
- `/sbatch training scripts/` - python scripts for hyperparam search, model training and a demo notebook.

## How to (supercomputing)

The necessary hyperparameters for the corresponding dataset may be determined by running:
 
- `sbatch supercomp`

Thereafter, the optimal architechtecture is used in `/sbatch training scripts/model_training.py`. The supercomp file should be modified to reference the current python script.
