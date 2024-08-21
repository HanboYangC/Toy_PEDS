#!/bin/bash
#SBATCH --job-name=PEDS_test      # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yangh@ufl.edu     # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=5
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --mem=10gb                     # Job memory request
#SBATCH --time=20:00:00               # Time limit hrs:min:sec
#SBATCH --output=./output/serial_test_%j.log   # Standard output and error log
pwd; hostname; date

module load conda
conda activate PEDS

cd /blue/guo/yangh/Toy_PEDS/
python 'search_weight_test_LF=16.py'


