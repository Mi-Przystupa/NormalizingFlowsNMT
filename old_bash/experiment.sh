#!/bin/bash
#SBATCH --time=0-12:00
#SBATCH --account=rrg-mageed
#SBATCH --mem=16000M
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1


#module load miniconda3
#source activate py36

#source activate py36

PYTHON=/scratch/anaconda3/bin/python


inputs="--use_bpe -src_bpe ./bpe_models/$5.model  -trg_bpe ./bpe_models/$6.model -word_dropout 0.0"
echo "Begining Experiment for $1 -> $2 translation with $3 number of flows"

echo "here are the inputs to the python script:"
echo "--source $1 --target $2 --model_type vnmt --use_flows --num_flows $3 --flow_type $4 $inputs"
$PYTHON main.py --source $1 --target $2 --model_type vnmt --use_flows --num_flows $3 --flow_type $4 $inputs 
