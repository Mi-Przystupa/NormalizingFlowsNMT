#!/bin/bash
#SBATCH --time=0-09:00
#SBATCH --account=rrg-mageed
#SBATCH --mem=96000M
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1


#set up stuff
module load miniconda3

module load miniconda3
conda env list
source activate py36
pwd

EPOCHS=5

i=$1 # epoch to run
out=$2

echo "params $i $EPOCHS $outfile"


#START OF EXPERIMENT+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
src=en
trg=de
lang_pair_config="--source $src --target $trg --dataset tabular --max_len 50"

vocab_config="--custom_vocab_src ./.data/$src.json --custom_vocab_trg ./.data/$trg.json --on_whitespace"

model_config="--model_type vnmt --hidden_size 1000 --num_layers 1 --emb_size 620 --z_dim 2000 --init_type normal --use_projection"

num_flows=0 #$1
flow_type=planar

flow_config="--num_flows $num_flows --flow_type $flow_type"

train_config="--epochs 1 --batch_size 80 --print_every 10000"

optimizer_config="--optimizer clippedadadelta --rho 0.95 --lr 1.0 --clip_norm 1.0"

elbo_config="--elbo_type MeanFieldELBO --num_particles 1 --kl_anneal 150000"

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

echo "starting job"
python main.py $lang_pair_config $vocab_config $model_config $flow_config $train_config $optimizer_config $elbo_config $continue_config $decode_config


#recursively resubmit job after finishing epoch
#./submit_epoch.sh $((i + 1))

#recursively resubmit job after finishing epoch
if [[ "$i" -le $EPOCHS ]] 
then
	echo "Going to submit next epoch"

	outfile="$out"-"$((i + 1))" # out file name
	sbatch --output "$outfile.out" vnmt_experiment.sh $((i + 1)) $out
fi



#BELOW: stuff I use to run on my desktop
#source activate py36
#python main.py $lang_pair $model_config $flow_config $train_config $optimizer_config $elbo_config
