#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00          # walltime

python infer360.py Kitchen new_shifted-disparity.png shifted_t.png Kitchen
# python infer360.py 'Kitchen/model_Y' 'KT_disp_by_depth.png' 'KT-t.png' 'Kitchen_model_Y'