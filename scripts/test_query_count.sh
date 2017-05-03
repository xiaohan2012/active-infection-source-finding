#! /bin/zsh

python query_count.py  -n 10 \
       -t grid --size_param 2-6 \
       -p 0.5 -q 0.1 \
       -e 0.2 \
       -m max_mu \
       --mwu_reward_method exact \
       -d
       
       
