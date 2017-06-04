#! /bin/zsh
# -g grid --param '2-6' \
# -g p2p-gnutella08 --param "" \
kernprof -l  paper_experiment.py \
  -g grid --param '2-6' \
  -m mst  \
  -l si \
  -p 0.5 \
  -q 1.0 \
  -o outputs/paper_experiment/test.pkl \
  -k 10 \
  -v

