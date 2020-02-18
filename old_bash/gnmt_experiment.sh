#!/bin/bash
#SBATCH --time=0-09:00
#SBATCH --account=rrg-mageed
#SBATCH --mem=96000M
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1

#PYTHON=/home/przy/projects/rrg-mageed/MT/code/anaconda3/bin/python

#set up stuff
module load miniconda3
source activate py36
pwd

EPOCHS=5

#Variables that need to be passed in 
i=$1 # epoch to run
out=$2

echo "params $i $EPOCHS $outfile"

#START OF EXPERIMENT+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
src=en
trg=de
lang_pair_config="--source $src --target $trg --dataset tabular --max_len 50"

#comments from paper: 
#they played with dropout
#WMT: 40% En- De, 50% De - En
#IWSLT 50% for both directions
#dropout applied to word embeddings on input to output projection layer (penultimate layer?)

vocab_config="--custom_vocab_src ./.data/$src.json --custom_vocab_trg ./.data/$trg.json --on_whitespace"

model_config="--model_type vaenmt --hidden_size 256 --num_layers 1 --emb_size 256 --z_dim 256 --init_type normal --use_projection"

num_flows=0 #$1
flow_type=planar

flow_config="--num_flows $num_flows --flow_type $flow_type"

#notes on optimizer:
#they also check performance by number of batches, where as I'm doing it by epoch :/ 
#probably fairly simple to add that...
#config authors used on the IWSLT and WMT datasets
#--optimizer adam --lr .0003 --batch_size 64 

train_config="--epochs 1 --batch_size 80 --print_every 10000 --dropout .40"
optimizer_config="--optimizer clippedadam --lr 0.0003 --clip_norm 1.0"


#comments from paper:
#they experiment with KL annealling from 20,000 - 80,000 steps
#results suggest 80,000
elbo_config="--elbo_type MeanFieldELBO --num_particles 1 --kl_anneal 80000"

#VERY IMPORTANT: YOU NEED TO SPECIFY EPOCH FOR THIS TO RUN RECURSIVELY
i=$1
continue_config="--load_epoch $i"
decode_config="--decode_alg beamsearch -k 10"


echo $lang_pair
echo $vocab_config
echo $model_config
echo $flow_config
echo $train_config
echo $optimizer_config
echo $elbo_config
echo $continue_config
echo $decode_config

echo "All the parameters"
echo "$lang_pair_config $vocab_config $model_config $flow_config $train_config $optimizer_config $elbo_config $continue_config $decode_config"
pwd
python --version
python3 --version

echo start job
python3 main.py $lang_pair_config $vocab_config $model_config $flow_config $train_config $optimizer_config $elbo_config $continue_config $decode_config


#recursively resubmit job after finishing epoch
if [[ "$i" -le $EPOCHS ]] 
then
	echo "Going to submit next epoch"
	outfile="$out"-"$((i + 1))"
 	#sbatch --output "$outfile.out" gnmt_experiment.sh $i
	 sbatch --output "$outfile.out" gnmt_experiment.sh $((i + 1)) $out
fi


#BELOW: stuff I use to run on my desktop
#source activate py36
#python main.py $lang_pair $model_config $flow_config $train_config $optimizer_config $elbo_config
