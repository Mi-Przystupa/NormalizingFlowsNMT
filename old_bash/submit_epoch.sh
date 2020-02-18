#!/bin/bash

#for a new experiment, specify -1 with file
#e.g. ./submit_epoch.sh -1 

#The input is the epoch to load...so if you try to do it before one exists...
#well it won't work

#inputs
# $1 experiment bash script to run e.g. gnmt_experiment.sh
# $2 epoch to run
# $3 name of outfile

#constant you set
EPOCHS=5

#variables you pass in 
experiment=$1 #bash script to run 
i=$2 # epoch to run
outfile="$3"-"$i" # out file name

echo "params $i $EPOCHS $outfile"
if [[ "$i" -le $EPOCHS ]] 
then
	echo "Going to submit next epoch"
 	#sbatch --output "$outfile.out" gnmt_experiment.sh $i
	 sbatch --output "$outfile.out" $experiment $i
fi
