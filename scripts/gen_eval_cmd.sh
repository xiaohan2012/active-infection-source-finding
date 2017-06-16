#! /bin/zsh

# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email dblp-collab)
# gtypes=(p2p-gnutella08 arxiv-hep-th enron-email)
# gtypes=(arxiv-hep-th enron-email)
gtypes=(p2p-gnutella08)
# gtypes=(pokec)
# gtypes=(dblp-collab slashdot twitter gplus pokec)
# gtypes=("barabasi-64")
# methods=(tbfs mst no-order)
methods=(tbfs mst no-order)
# methods=(tbfs)
# models=("si")
models=("si" "ct")
# models=("ct")

first="${gtypes[1]}"

if [ "${first}" = "barabasi-64" ]; then
    qs=(0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0)
else
    if [ "${first}" = "dblp-collab" ]; then
	qs=(0.1)
    else
	qs=(0.005 0.01 0.015 0.02 0.025 0.03 0.035 0.04 0.045 0.05 0.055 0.06 0.065 0.07 0.075 0.08 0.085 0.09 0.095 0.1)
    fi
fi

for gtype in $gtypes; do
    for method in $methods; do
	for model in $models; do
	    model=("${(@s/ /)model}")
	    model=${model[1]}

	    print "python evaluate.py -g ${gtype}  -m ${method}  -l ${model}  -q ${qs}"
	done
    done
done
