#! /bin/zsh

# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email dblp-collab)
gtypes=(p2p-gnutella08 arxiv-hep-th enron-email)
methods=(no-order)
model_params=("si -p 0.1" "ic -p 0.5" "sp -p 0.1")
qs=(0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09 0.095 0.1)

for gtype in $gtypes; do
    for method in $methods; do
	for model_param in $model_params; do
	    for q in $qs; do
		model=("${(@s/ /)model_param}")
		model=${model[1]}
		print "python paper_experiment.py -g ${gtype}  -m ${method}  -l ${model_param}  -q ${q}  -o outputs/paper_experiment/${gtype}/${model}/${method}/qs/${q}.pkl -k 100"
	    done
	done
    done
done
