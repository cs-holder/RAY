#!/bin/bash
#SBATCH -J exp
#SBATCH --time=72:00:00     # walltime
#SBATCH -o experiment.out
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH --gres=gpu:a100-sxm4-80gb:1

# a100-sxm-64gb
# nvidia_a100_80gb_pcie

source ~/.bashrc
conda activate pyg

export PYTHONPATH=`pwd`
echo $PYTHONPATH

ARGS=${@:1}

cmd="python3 -m src.main \
    $ARGS"

echo "Executing $cmd"

$cmd