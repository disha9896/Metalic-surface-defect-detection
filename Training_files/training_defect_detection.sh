#!/bin/bash -l

#SBATCH --job-name=defect_detection      # Name of your job
#SBATCH --time=1-00:00:00               # Time limit, format is Days-Hours:Minutes:Seconds

#SBATCH --output=%x_%j.out              # Where to save output
#SBATCH --error=%x_%j.err               # Where to save error messages

#SBATCH --ntasks=1                      # 1 task (default of 1 CPU per task)
##SBATCH --mem=4g                        # 4GB of RAM for the whole job
#SBATCH --gres=gpu:a100:1                 # Request 1 P4 GPU

#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=10g

#SBATCH --account=core-id            # Slurm account
#SBATCH --partition=tier3             # Partition to run on

#SBATCH --mail-user=slack:@ds2467
#SBATCH --mail-type=ALL

# Load your software here
spack load /arwp2px cuda@12.4.0
conda activate defect_detection

# Your code here
python3 -u Yolov8_seg.py
