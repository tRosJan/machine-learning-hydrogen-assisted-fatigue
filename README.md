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


## Citation: 
If you use this code in your research, please cite the following paper:

@article{aduwenye2025application,
  title={Application of machine learning for predicting hydrogen-assisted fatigue crack growth},
  author={Aduwenye, Presley and {\v{S}}ebek, Franti{\v{s}}ek and Yeddu, Hemantha Kumar},
  journal={Computational Materials Science},
  volume={260},
  pages={114207},
  year={2025},
  publisher={Elsevier}
}

Paper: Application of machine learning for predicting hydrogen-assisted fatigue crack growth, Computational Materials Science, Volume 260, Article 114207 (2025).

Citing the original publication helps acknowledge the research behind this implementation and supports the continued development and dissemination of open-source scientific tools.
