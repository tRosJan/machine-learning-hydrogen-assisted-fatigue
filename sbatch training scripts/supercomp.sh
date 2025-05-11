#!/bin/bash -l
#SBATCH --job-name=grid                   # Job name
#SBATCH --error=%j.err                    # Standard error file
#SBATCH --output=%j.out                   # Standard output file 
#SBATCH --nodes=3                         # Number of nodes
#SBATCH --ntasks=1                        # Number of tasks 
#SBATCH --cpus-per-task=20                # Number of CPU cores per task
#SBATCH --gres=gpu:v100:2                  # gpu
#SBATCH --time=72:00:00                   # time
#SBATCH --partition=gpu                   # Partition to run the job on (GPU)
#SBATCH --mem-per-cpu=8G                  # Memory per CPU core
#SBATCH --account=<rem>     #account here

# Load necessary modules
module load python-data                # Load Python (adjust version as per environment)
module load tensorflow            # Load TensorFlow (GPU-enabled, adjust version as per environment)
module load cuda                   # Load CUDA (adjust version based on TensorFlow compatibility)

# Navigate to your script directory
cd /directory

# Run the Python script
python <scriptname>

