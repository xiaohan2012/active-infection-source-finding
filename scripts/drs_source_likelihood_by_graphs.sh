#! /bin/zsh
parallel \
    python plot_source_likelihood_modeling_drs_by_ref_nodes_fraction.py \
    ::: kr-rand kr-peri kr-hier balanced-tree er barabasi

