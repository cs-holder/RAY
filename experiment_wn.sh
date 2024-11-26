#!/bin/bash
#SBATCH -J exp
#SBATCH --time=32:00:00     # walltime
#SBATCH -o experiment.out
#SBATCH -p compute                            
#SBATCH -N 1                                  
#SBATCH --gres=gpu:a100-sxm-40gb:1

source ~/.bashrc
conda activate pyg

export PYTHONPATH=`pwd`
echo $PYTHONPATH

ARGS=${@:1}

cmd="python3 -m src.main \
    $ARGS"

echo "Executing $cmd"

$cmd