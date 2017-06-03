#! /bin/zsh
# -g grid --param '2-6' \
# -g p2p-gnutella08 --param "" \
kernprof -l  paper_experiment.py \
  -g p2p-gnutella08 --param "" \
  -m greedy  \
  -l si \
  -p 0.5 \
  -q 0.01 \
  -o outputs/paper_experiment/test.pkl \
  -k 10 \
  -v

