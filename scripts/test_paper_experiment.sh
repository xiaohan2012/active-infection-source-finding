#! /bin/zsh

kernprof -l  paper_experiment.py \
       -g grid --param '2-6' \
       -m greedy  \
       -l si \
       -p 0.5 \
       -q 0.8 \
       -o outputs/paper_experiment/test.pkl \
       -k 10 \
       -v

