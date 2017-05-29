#! /bin/zsh

python source_likelihood_estimation_experiment.py \
       -g balanced-tree -p '2-6' \
       -m region_mst  \
       --n1 10 --n2 10 \
       --p1 0.5 --p2 0.5 \
       --q1 1.0 --q2 1.0 \
       -d
