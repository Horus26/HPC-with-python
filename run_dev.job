#!/bin/bash -x
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:30:00
#SBATCH -J HPC_WITH_PYTHON
#SBATCH --mem=6gb
#SBATCH --output=HPC_OUTPUT.out
#SBATCH --error=HPC_ERROR.err
#SBATCH --export=ALL
#SBATCH --partition=dev_multiple

module load devel/python/3.10.0_gnu_11.1
module load compiler/gnu/12.1
module load mpi/openmpi/4.1

module list

echo "Running on ${SLURM_JOB_NUM_NODES} nodes with ${SLURM_JOB_CPUS_PER_NODE} cores each."
echo "Each node has ${SLURM_MEM_PER_NODE} of memory allocated to this job."
time mpirun python3 Milestone7_Parallelization_Dev.py