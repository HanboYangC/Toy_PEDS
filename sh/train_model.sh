#!/bin/sh
#SBATCH --job-name=DrugCLIP_test   # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yangh@ufl.edu   # Where to send mail	
#SBATCH --nodes=1                     # Use one node
#SBATCH --ntasks=1                    # Run a single task
#SBATCH --cpus-per-task=10            # Use 1 core
#SBATCH --partition=gpu
#SBATCH --gpus=geforce:1
#SBATCH --mem=10gb                   # Memory limit
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --output=output/serial_test_%j.out   # Standard output and error log

pwd; hostname; date

module load conda
conda activate PEDS

cd "/blue/guo/yangh/Toy_PEDS"

export PYTHONPATH="${PYTHONPATH}:/blue/guo/yangh/Toy_PEDS"

python ./tools/train_model.py