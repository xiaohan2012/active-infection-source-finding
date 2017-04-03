#! /bin/zsh

params=("pl-tree 2-6"
	"balanced-tree 2-6"
	"er 2-6"
	"barabasi 2-6"
	"kr-peri 10-6"
	"kr-hier 10-6"
	"kr-rand 10-6")

for param in $params; do
    eval "python how-well-can-we-model-probability.py $param"
done
