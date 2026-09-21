#!/bin/bash
##----------------------- Start job description -----------------------
#SBATCH --partition=standard
#SBATCH --job-name=generate_result
#SBATCH --ntasks=1
#SBATCH --time=05:40:00
#SBATCH --mem-per-cpu=24G
#SBATCH --mail-type=END,FAIL,STAGE_OUT
#SBATCH --mail-user=grmapal2@gmail.com
#SBATCH --chdir=/home/v582/v582962/Pablo/ViNet/ViNet/output/
##------------------------ End job description ------------------------

module purge && ml MATLAB/2022b

cd "/home/v582/v582962/Pablo/ViNet/ViNet/code_for_Metrics"

srun matlab -r "run('Demo.m'); exit;"
