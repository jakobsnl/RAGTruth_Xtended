#!/bin/bash
#SBATCH --cpus-per-task=8
#SBATCH --partition=h100-ferranti
#SBATCH --gres=gpu:2
#SBATCH --nodes=1
#SBATCH --mem=64G
#SBATCH --time=3-00:00
#SBATCH -o /home/oh/owl777/logs/%j_%x.out

# Start code inside a singularity container
#singularity exec --bind /:/host --nv RAGTruth_analysis.sif bash \

bash -c "
source ~/.bashrc
conda  activate /home/oh/owl777/.conda/envs/rtx
python rtx/create_dataset.py --input_dir dataset/RAGTruth/ --save_dir dataset/rtx --add_logits --add_hidden_states"