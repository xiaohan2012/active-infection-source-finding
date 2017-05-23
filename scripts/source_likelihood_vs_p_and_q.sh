#! /bin/zsh

other_params="--cache_sim --n2 100"
gtypes=(balanced-tree grid barabasi er)

exps=(6)

# methods=("-m steiner-tree"
#     "-m time-order -c and"
#     "-m time-diff -c and")
methods=("-m tbfs")

for method in $methods; do
    for exp in $exps; do
	for gtype in $gtypes; do
	    print "$gtype $exp $method"
	    eval "python source_likelihood_estimation_experiment.py -g $gtype -p 2-$exp $method $other_params"
	done
    done
done
