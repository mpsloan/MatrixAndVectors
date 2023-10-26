#!/bin/bash

#SBATCH -N 1
#SBATCH --account=pwsu0495
#SBATCH --tasks-per-node=28
#SBATCH --job-name=proj04
#SBATCH --output=proj04.txt

# variables for matrix runs
L2=100000

# this gets casted to an int in my program
# so after decimal will get truncated
B=$(echo $L2*2.4 | bc)

module load mvapich2
mpicc -o proj04.mpi -O2 proj04b.c
mpiexec -n 28 ./proj04.mpi 100000 $L2 1 $B 45 135


