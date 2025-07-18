#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00
#SBATCH -o /home/oh/owl777/logs/%j_%x.out

bash -c "
source ~/.bashrc
conda  activate /home/oh/owl777/.conda/envs/ragtruth
python rtx/analyse_hallucination.py --output_dir results/ --dataset_dir dataset/rta/ --sequence_scopes all first second third+ 
"