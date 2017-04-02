#! /bin/zsh

params=("pl-tree 2-6" "balanced-tree 2-6" "barabasi 2-6" "kr-peri 10-8" "kr-hier 10-8" "kr-rand 10-8" "p2p-gnutella08 ''")

for param in $params; do
    eval "python how-well-can-we-model-probability.py $param"
done
