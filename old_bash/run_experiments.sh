#!/bin/bash


#Bash script to run all experiments (could be changed to submit as jobs)

LANGS=("de-en" "ar-en")
num_flows=(0 1 2 4 8 16 32)
flow_types=("iaf" "planar" )
bpe_models=("german de_english" "arabic ar_english")

#outdir=./logs/ # use this for training
outdir=./logs/
for f_t in ${flow_types[@]}
do
    i=0
    for entry in ${LANGS[@]}
    do
        bpe_model=${bpe_models[$i]}
        i=$((i+1))
        for n_f in ${num_flows[@]}
        do
            L1=$(echo $entry |rev| cut -d '-' -f 2 | rev)
            L2=$(echo $entry |rev| cut -d '-' -f 1 | rev)	

            echo "Inputs to experiment: $L1 $L2 $n_f $f_t $bpe_model"
            #./experiment.sh $L1 $L2 $n_f $f_t $bpe_model >> "$outdir""$L1"-"$L2"-num-flows-"$n_f"-"$f_t".txt
            ./eval_models.sh $L1 $L2 $n_f $f_t $bpe_model >> "$outdir"validationres-"$L1"-"$L2"-num-flows-"$n_f"-"$f_t".txt

        done
    done
done

