#!/bin/bash
###
#SBATCH --job-name=download-openvid

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=1000GB
#SBATCH --time=72:00:00
#SBATCH --output="logs/%x-%j.out"
#SBATCH --exclusive

# Log the assigned nodes
echo "START TIME: $(date)"
echo "Using nodes: $SLURM_JOB_NODELIST"


source /ml-tests/OpenVid-1M/.env/bin/activate

srun python \
    unpack-openvid.py


echo "END TIME: $(date)"