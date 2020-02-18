#!/bin/bash
#SBATCH --time=0-05:00
#SBATCH --account=rrg-mageed
#SBATCH --mem=96000M
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1


#set up stuff
module load miniconda3
conda env list
source activate py36
pwd

EPOCHS=0

i=$1 # epoch to load, if greater than 0
out=$2

#knobs to turn
model=$3
kl_anneal=$4
z_dim=$5
#word_drop=$6

#Flows information

flow_type=$6
num_flows=$7 #$1
num_particles=$8

echo "params $EPOCHS $i $out $model $kl_anneal $z_dim"

#START OF EXPERIMENT+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
src=de
trg=en
#lang_pair_config="--source $src --target $trg --dataset tabular --max_len 50" #WMT2014 setup
lang_pair_config="--source $src --target $trg --dataset IWSLT --max_len 50"

#below was for wmt14 dataset
#vocab_config="--custom_vocab_src ./.data/$src.json --custom_vocab_trg ./.data/$trg.json --on_whitespace"
vocab_config="--src_bpe ./.data/bpe_models/german.model --trg_bpe ./.data/bpe_models/de_english.model --use_bpe"

#original model config for vnmt
#model_config="--model_type vnmt --hidden_size 1000 --num_layers 1 --emb_size 620 --z_dim 2000 --init_type normal --use_projection"
#roughly original for vaenmt
#model_config="--model_type vaenmt --hidden_size 256 --num_layers 1 --emb_size 256 --z_dim 256 --init_type normal --use_projection"
model_config="--model_type $model --hidden_size 256 --num_layers 1 --emb_size 256 --z_dim $z_dim --init_type normal --use_projection"

flow_config="--num_flows $num_flows --flow_type $flow_type --use_flows"

#original vnmt used batch_size = 80 and no dropout...but dropout helps generalize as do smaller batches 
train_config="--epochs 0 --batch_size 64 --print_every 1000 --dropout .50 -opt validate" #vnmt technically did not use dropout, but it does help with generalization 

#optimizer_config="--optimizer clippedadadelta --rho 0.95 --lr 1.0 --clip_norm 1.0" #config from original vnmt
optimizer_config="--optimizer clippedadam --lr 0.0003 --clip_norm 1.0"

elbo_config="--elbo_type MeanFieldELBO --num_particles $num_particles --kl_anneal $kl_anneal --to_anneal q_p --word_dropout 0.1"

#VERY IMPORTANT: YOU NEED TO SPECIFY EPOCH FOR THIS TO RUN RECURSIVELY
continue_config="--load_epoch $i --load_latest_epoch"
decode_config="--decode_alg beamsearch -k 10 --length_norm" #length norm helps...and hopefully just works (fingers crossed)

echo $lang_pair_config
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

echo "starting job"
python main.py $lang_pair_config $vocab_config $model_config $flow_config $train_config $optimizer_config $elbo_config $continue_config $decode_config


#recursively resubmit job after finishing epoch
#./submit_epoch.sh $((i + 1))

#BELOW: stuff I use to run on my desktop
#source activate py36
#python main.py $lang_pair_config $model_config $flow_config $train_config $optimizer_config $elbo_config
