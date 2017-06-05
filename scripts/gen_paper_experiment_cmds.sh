#! /bin/zsh

# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email dblp-collab)
# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email)
gtypes=(p2p-gnutella08)
# gtypes=("grid-64")
# gtypes=("grid-64")
# methods=(mst tbfs no-order greedy)
methods=(mst tbfs no-order)
# model_params=("si -p 0.1" "ic -p 0.5" "sp -p 0.1")
# model_params=("si -p 0.1" "ic -p 0.083" "sp -p 0.1")
# model_params=("ic -p 0.083")  #
# model_params=("ic -p 0.2660444431189779")  #  grid
# p2p: 0.035241715776066926
model_params=("ic -p 0.035241715776066926")
# arxiv: 0.03222184517702736
# enron: 0.00844468246106171

qs=(0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09 0.095 0.1)
# qs=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)

for gtype in $gtypes; do
    for method in $methods; do
	for model_param in $model_params; do
	    for q in $qs; do
		# temporary
		
		model=("${(@s/ /)model_param}")
		model=${model[1]}

		# add command if not computed
		check_path="outputs/paper_experiment/${gtype}/${model}/${method}/qs/${q}/99.pkl"
		if [[ ! -a ${check_path} ]]; then
		    print "python paper_experiment.py -g ${gtype}  -m ${method}  -l ${model_param}  -q ${q}  -o outputs/paper_experiment/${gtype}/${model}/${method}/qs/${q}.pkl -k 100"
		else
		    print "python paper_experiment.py -g ${gtype}  -m ${method}  -l ${model_param}  -q ${q}  -o outputs/paper_experiment/${gtype}/${model}/${method}/qs/${q}.pkl -k 100"
		fi
		
	    done
	done
    done
done
