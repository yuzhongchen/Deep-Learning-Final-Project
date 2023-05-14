#!/bin/bash 
#SBATCH --nodes=1 # requests n compute servers
#SBATCH --ntasks-per-node=1 # runs n tasks on each server
#SBATCH --cpus-per-task=20 # uses n compute core per task
#SBATCH --gres=gpu:rtx8000:4
#SBATCH --time=48:00:00
#SBATCH --mem=200GB
#SBATCH --job-name=train_div
#SBATCH --output=egopat3d.out

module purge
export LD_LIBRARY_PATH=./../anaconda3/lib/:$LD_LIBRARY_PATH
source /scratch/yc6317/anaconda3/etc/profile.d/conda.sh
conda activate egopat3d
python ./train_two.py --data_path ./dataset
# python ./test_two.py --data_path ./dataset.hdf5 --model_name two --checkpoint ./LSTM
# python ./eval.py --data_path ./Dataset --model_name two