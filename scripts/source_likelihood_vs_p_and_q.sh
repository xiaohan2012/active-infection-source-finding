#! /bin/zsh

gtypes=(kr-rand kr-peri kr-hier balanced-tree er barabasi)
# gtypes=(barabasi)
# exps=(6 7 8 9)
exps=(6)
# methods=(1st 1st_time drs)
# methods=(drs)
# methods=(drs_time_early drs_time_late drs_time_mean)
# methods=(pair_order)
# methods=(time_diff)
methods=(drs pair_order time_diff_dist_sum_quad time_diff_dist_diff_quad time_diff_dist_diff_abs)

for method in $methods; do
    for exp in $exps; do
	for gtype in $gtypes; do
	    print "$gtype $exp $method"
	    eval "python source_likelihood_estimation_experiment.py $gtype 2-$exp $method"
	done
    done
done
