#!/bin/bash
#SBATCH --job-name="tweet_assignment"
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH -o final_2n8c.txt

module load foss/2022a
module load Python/3.10.4
module load mpi4py/3.1.4
module load SciPy-bundle/2022.05

time mpiexec -n 8 python3 final_model.py

##DO NOT ADD/EDIT BEYOND THIS LINE##
##Job monitor command to list the resource usage
my-job-stats -a -n -s