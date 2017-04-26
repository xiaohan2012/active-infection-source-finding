#! /bin/zsh

gtypes=(kr-rand kr-peri kr-hier balanced-tree er barabasi)

exp=6

for gtype in $gtypes; do
    eval "python convert_graphml_to_gt.py $gtype/2-$exp"
done
