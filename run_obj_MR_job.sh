#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00          # walltime

python infer360.py Meeting new_shifted-disparity.png shifted_t.png Meeting
# python infer360.py 'Meeting/model_Y' 'MR_disp_by_depth.png' 'MR-t.png' 'Meeting_model_Y'