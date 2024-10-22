#!/bin/bash

#SBATCH --time=60:00:00
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=4000
#SBATCH --gpus=1
#SBATCH --gres=gpumem:16G

module purge
module load stack eth_proxy python_cuda
git checkout deteriorated_ic
source venv/bin/activate

accelerate launch --config_file ../swin-transformer-pde/configs/accelerate/single-gpu.yaml scOT/train.py --config configs/run.yaml --wandb_run_name case2 --wandb_project_name scOT-sparse --max_num_train_time_steps 14 --train_time_step_size 1 --data_path /cluster/scratch/herdem/ --checkpoint_path /cluster/work/math/herdem/checkpoints/