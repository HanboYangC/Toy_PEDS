#!/bin/bash
#SBATCH --job-name=run.sh      # Job name
#SBATCH --mail-type=ALL          # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=yangh@ufl.edu     # Where to send mail
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=3
#SBATCH --mem=10gb                     # Job memory request
#SBATCH --time=30:00:00               # Time limit hrs:min:sec
#SBATCH --h3=./h3/serial_test_%j.log   # Standard h3 and error log
pwd; hostname; date

