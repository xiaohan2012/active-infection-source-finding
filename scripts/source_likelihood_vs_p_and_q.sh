#! /bin/zsh

other_params="--cache_sim --n2 128"
gtypes=(balanced-tree grid barabasi er)

exps=(6)

methods=("-m steiner"
    "-m order -c and"
    "-m dist -c and")

for method in $methods; do
    for exp in $exps; do
	for gtype in $gtypes; do
	    print "$gtype $exp $method"
	    eval "python source_likelihood_estimation_experiment.py -g $gtype -p 2-$exp $method $other_params"
	done
    done
done
