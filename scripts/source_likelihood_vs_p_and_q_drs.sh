#! /bin/zsh

gtypes=(kr-rand kr-peri kr-hier balanced-tree er barabasi)
exps=(6)

for exp in $exps; do
    for gtype in $gtypes; do    
	eval "python how-well-can-we-model-probability-drs.py $gtype 2-$exp"
    done
done
