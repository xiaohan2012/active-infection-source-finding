#! /bin/zsh

# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email dblp-collab)
# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email)
# gtypes=(p2p-gnutella08)
# gtypes=(arxiv-hep-th)
gtypes=(enron-email)
# gtypes=(dblp-collab slashdot twitter gplus)
# gtypes=(arxiv-hep-th enron-email)
# gtypes=("barabasi/2-6/" "barabasi/2-7/" "barabasi/2-8/" "barabasi/2-9/"
# 	"barabasi/2-10/" "barabasi/2-11/" "barabasi/2-12/" "barabasi/2-13/"
# 	"barabasi/2-14/")
# gtypes=("barabasi-64")
# gtypes=("grid-64" "barabasi-64")
# gtypes=("grid-64")
# methods=(mst tbfs no-order greedy)
methods=(mst tbfs no-order)
# methods=(mst)
model_params=("si -p 0.1" "ct -p 0.0")  # barabasi

# model_params=("ic -p 0.0831108693813754")  # barabasi
# model_params=("ic -p 0.2660444431189779")  #  grid
# model_params=("ic -p 0.035241715776066926")  #  p2p
# model_params=("ic -p 0.03222184517702736")  #  arxiv
# model_params=("ic -p 0.00844468246106171")  #  enron

# model_params=("si -p 0.1")
# methods=(mst tbfs no-order)

if [[ "${gtypes[1]}" = "barabasi-64" || "${gtypes[1]}" = "grid-64" ]]; then
    qs=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
    methods=(mst tbfs no-order)
else
    if [ "${gtypes[1]}" = "pokec" ]; then
	qs=(0.1)
	methods=(tbfs)
	model_params=("si -p 0.1")
    else
	qs=(0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09 0.095 0.1)
    fi
    
fi

# model_params=("si -p 0.1")
# qs=(0.1)
# methods=(tbfs)

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
		    print "python paper_experiment.py -g ${gtype}  -m ${method}  -l ${model_param}  -q ${q}  -o outputs/paper_experiment/${gtype}/${model}/${method}/qs/${q}.time.pkl -k 100"
		else
		    print "python paper_experiment.py -g ${gtype}  -m ${method}  -l ${model_param}  -q ${q}  -o outputs/paper_experiment/${gtype}/${model}/${method}/qs/${q}.pkl -k 100"
		    # print done
		fi
		
	    done
	done
    done
done
