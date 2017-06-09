#! /bin/zsh
# -g grid --param '2-6' \
# -g p2p-gnutella08 --param "" \
# -g pokec --param "" \
kernprof -l  paper_experiment.py \
  -g grid --param '2-6' \
  -m tbfs  \
  -l si \
  -p 0.1 \
  -q 0.1 \
  -o outputs/paper_experiment/test.pkl \
  -k 10 \
  -v

