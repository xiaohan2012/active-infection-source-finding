#! /bin/zsh

graphs=(barabasi  er  grid  kr-hier  kr-peri  kr-rand  pl-tree)
for graph in $graphs; do
    python plot_query_count_vs_graph_size.py $graph
done
