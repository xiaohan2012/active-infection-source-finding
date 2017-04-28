#! /bin/zsh

gtypes=(arxiv-hep-th p2p-gnutella08)

methods=("-m exact"
	 "-m exact -c and"
	 "-m order -c and"
	 "-m order -c or"
	 "-m dist -c and")

for gtype in $gtypes; do
    for method in $methods; do
	print "$gtype $method"
	eval "python source_likelihood_estimation_experiment.py -g $gtype -p '' $method  --n1 100 --n2 100 --p1 0.5 --p2 0.5 --q1 0.01 --q2 0.01"
	# eval "python source_likelihood_estimation_experiment.py -g $gtype -p 2-$exp $method"
	# 
    done
done

