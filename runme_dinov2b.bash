#!/bin/bash
#SBATCH --account punim0980 
#SBATCH --partition gpu-a100-short

#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=112G
#SBATCH --time=1:00:00
#SBATCH --output runme_dinov2b.out
#SBATCH --open-mode=truncate
#SBATCH --job-name=runme_dinov2b
#SBATCH --mail-type=ALL
#SBATCH --mail-user=nathaniel.bloomfield@unimelb.edu.au

# sinteractive --account=punim0980 --time=240 --ntasks=1 --mem=112gb --cpus-per-task 8 -p gpu-a100-short --gres=gpu:1
# punim0980


export mysystem=spartan
export env_seed=1
export HYDRA_FULL_ERROR=1
export TORCH_HOME=/data/cephfs/punim0980/torch_hub
export TRANSFORMERS_CACHE=/data/cephfs/punim0980/torch_hub
export HUGGINGFACE_HUB_CACHE=/data/cephfs/punim0980/torch_hub/hub
export LIGHTLY_DID_VERSION_CHECK=True

module load Python/3.10.4
module load CUDA/11.7.0
source /data/cephfs/punim0980/venvs/mysemiv4/bin/activate

cd /data/cephfs/punim0980/cloned_repos/funnybirds-framework-comfe

python evaluate_explainability.py --data /data/cephfs/punim0980/data/FunnyBirds --model comfe_dinov2_wreg --explainer comfe_dinov2_wreg --accuracy --controlled_synthetic_data_check --target_sensitivity --single_deletion --preservation_check --deletion_check --distractibility --background_independence --gpu 0


