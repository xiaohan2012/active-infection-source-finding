#! /bin/zsh

python paper_experiment.py -g slashdot  -m tbfs  -l si -p 0.1  -q 0.1  -o outputs/paper_experiment/slashdot/si/tbfs/qs/0.1.time.pkl -k 100 --parallel
python paper_experiment.py -g twitter  -m tbfs  -l si -p 0.1  -q 0.1  -o outputs/paper_experiment/twitter/si/tbfs/qs/0.1.time.pkl -k 100 --parallel
python paper_experiment.py -g dblp-collab  -m tbfs  -l si -p 0.1  -q 0.1  -o outputs/paper_experiment/dblp-collab/si/tbfs/qs/0.1.time.pkl -k 100 --parallel
python paper_experiment.py -g gplus  -m tbfs  -l si -p 0.1  -q 0.1  -o outputs/paper_experiment/gplus/si/tbfs/qs/0.1.time.pkl -k 100 --parallel
