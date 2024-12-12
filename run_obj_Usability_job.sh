#!/bin/bash 

#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:60:00          # walltime

python infer360.py Usability new_shifted-disparity.png shifted_t.png Usability
# python infer360.py 'Usability/model_Y' 'UL_disp_by_depth.png' 'UL-t.png' 'Usability_model_Y'