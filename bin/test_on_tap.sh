#!/bin/bash

#for seed in 4 18 27 36 42
#do
  #grep -v sapiens ../data/evaluations/10-fold-cross-val/training_split_${seed}/tap.csv > ../data/evaluations/10-fold-cross-val/training_split_${seed}/tap_filt.csv
  #cat ../data/evaluations/10-fold-cross-val/training_split_${seed}/tap_filt.csv > ../data/evaluations/10-fold-cross-val/training_split_${seed}/tap.csv
  #rm ../data/evaluations/5x2cv/training_split_${seed}_a/tap.csv
  #rm ../data/evaluations/5x2cv/training_split_${seed}_b/tap.csv
#done

python3 ./test_on_tap.py --seed 4 --prepro 1
python3 ./test_on_tap.py --seed 18  --prepro 1
python3 ./test_on_tap.py --seed 27  --prepro 1
python3 ./test_on_tap.py --seed 36 --prepro 1
python3 ./test_on_tap.py --seed 42 --prepro 1
