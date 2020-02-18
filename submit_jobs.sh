#!/bin/bash

#Run this to submit all jobs of interest to compute canada

#out="gnmt"
#sbatch --out "$out".out gnmt_experiment.sh "-1" $out


MODELS=("mod_vaenmt_no_lm") # ("vanilla_joint_nmt") # ("mod_vnmt" "mod_vaenmt") #("vanilla_nmt") #("mod_vnmt" "mod_vaenmt") #"vanilla_nmt") # "mod_vaenmt" "mod_vnmt") # vnmt") #("mod_vaenmt" "vaenmt" "mod_vnmt" "vnmt") #("vnmt") # "vaenmt") #("vnmt" "vaenmt")
#MODELS=("vanilla_joint_nmt")
KL_ANNEAL=(80000) #(1 6200) #should roughly be 6152 to be exact... or 6152.625 if you want to be super exact
DIM_Z=(128 256) #(128 256) #128 256) #(1 2 8 64 256 512)
#DIM_Z=(0)
FLOWS=("cond-planar-v2" "cond-iaf") #("cond-planar" "cond-iaf") #("none" "planar" "iaf" "householder" sylvester")
#FLOWS=("cond-planar-v2")
#NUM_FLOWS=(0) 
NUM_FLOWS=(0 1 2 4 8 16) #(0 1 2 4 8 16)
NUM_PARTICLES=(10)
#NUM_PARTICLES=(1) #for vanilla models, means they'll go quicker, if BLEU score did go up with 0 stochastic units in the model...i don't know what to believe anymore
#init_epoch=-1
init_epoch=0 #this is for validate/test

for model in ${MODELS[@]}
do
	echo $model
    for kl in ${KL_ANNEAL[@]}
        do
		echo $kl
        for dim_z in ${DIM_Z[@]}
            do
			for flow_type in ${FLOWS[@]}
			do
				for num_flow in ${NUM_FLOWS[@]}
				do
					for num_particles in ${NUM_PARTICLES[@]}
					do
						out="./outfiles/$model-$kl-$dim_z-$flow_type-$num_flow-$num_particles"
						echo "Inputs to experiment: $init_epoch $out $model $kl $dim_z $flow_type $num_flow"
						#sbatch --out "$out-no-attn-init".out exp_no_attention.sh $init_epoch $out $model $kl $dim_z $flow_type $num_flow $num_particles
						sbatch --out "$out-no-attn-validate".out val_no_attention.sh $init_epoch $out $model $kl $dim_z $flow_type $num_flow $num_particles
						sbatch --out "$out-no-attn-test".out test_no_attention.sh $init_epoch $out $model $kl $dim_z $flow_type $num_flow $num_particles

						#sbatch --out "$out-init".out experiment.sh $init_epoch $out $model $kl $dim_z $flow_type $num_flow $num_particles
						#sbatch --out "$out-validate".out validate.sh $init_epoch $out $model $kl $dim_z $flow_type $num_flow $num_particles
						#sbatch --out "$out-test".out test.sh $init_epoch $out $model $kl $dim_z $flow_type $num_flow $num_particles

					done
				done
			done
         done
    done
done



