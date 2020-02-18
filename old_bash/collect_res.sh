#!/bin/bash


PYTHON=/home/przy/projects/rrg-mageed/MT/code/anaconda3/bin/python

dataset=dataset_IWSLT_de-en
$PYTHON main.py --combine_results $dataset

