#!/bin/bash
#
#SBATCH --job-name=benchmark_su3
#SBATCH --nodes=4
#SBATCH --time=1:30:00
#SBATCH --partition=qp3
#SBATCH --switches=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --workdir=/home/fim16418/slurm/grid/sub-NUMA_clustering

module load mpi/openmpi-x86_64

export OMP_NUM_THREADS=64
#export OMP_PLACES=threads  #thread placement
#export OMP_PROC_BIND=TRUE  #threads not moved between CPUs
#export GOMP_CPU_AFFINITY=0 #binds threads to CPUs

now="$(date +"%T")"
outputfile="benchmark_su3_grid3_$now.txt"

MPIENV="--report-bindings --map-by numa --bind-to numa --mca mtl ofi --mca pml cm"

thread_pinning_env=$(/bug/opt/ur/scripts/get_thread_pinning_env.sh gnu 4 scatter); $thread_pinning_env

echo "SLURM_JOB_ID=$SLURM_JOB_ID" > $outputfile

echo $now
echo $outputfile
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES"
echo "SLURM_NTASKS_PER_CORE=$SLURM_NTASKS_PER_CORE"
echo "SLURM_NTASKS_PER_NODE=$SLURM_NTASKS_PER_NODE"
echo "SLURM_NTASKS_PER_SOCKET=$SLURM_NTASKS_PER_SOCKET"
echo "SLURM_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK"
echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "MPIENV=$MPIENV"

echo "Starting job"

mpirun $MPIENV /home/fim16418/Grid3/Grid/build/benchmarks/Benchmark_su3 --nLoops 10000 --outFile test.txt --mpiLayout 2 2 2 2 > $outputfile

echo "Job done"
