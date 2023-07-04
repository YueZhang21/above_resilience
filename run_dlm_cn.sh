#!/bin/bash
#SBATCH --time=30:00:00             # total run time limit (HH:MM:SS)
#SBATCH --ntasks=1                  # 
#SBATCH --mem-per-cpu=4G
#SBATCH --array=0-174
#SBATCH --job-name=dlm_cn    # create a short name for your job
#SBATCH --account=PAS2094           # account name
#SBATCH --output=JobInfo/%x_%a.out  # out message
#SBATCH --error=JobInfo/%x_%a.err   # error message
#SBATCH --mail-type=ALL             # mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=zhang.12439@osu.edu

source /users/PAS2094/yuezhang/local/anaconda3/2021.11/etc/profile.d/conda.sh
module use $HOME/local/share/lmodfiles 
module load anaconda3/2021.11
conda activate pygdal36

python run_dlm_cn.py
