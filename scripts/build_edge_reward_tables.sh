#! /bin/zsh

graphs=(barabasi  er  grid  kr-hier  kr-peri  kr-rand  pl-tree)
for graph in $graphs; do
    for size_param in data/$graph/*-*/; do
	size_param=$(basename $size_param)
	print "python build_edge_reward_table.py ${graph} ${size_param}"
    done
done
