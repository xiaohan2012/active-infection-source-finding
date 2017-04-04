#! /bin/zsh

gtypes=(kr-rand kr-peri kr-hier balanced-tree er barabasi)
exps=(6 7 8 9)
method=1st

for exp in $exps; do
    for gtype in $gtypes; do
	print "$gtype $exp"
	eval "python how-well-can-we-model-probability.py $gtype 2-$exp $method"
    done
done
