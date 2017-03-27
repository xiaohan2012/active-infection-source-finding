#! /bin/zsh

graphs=(barabasi  er  grid  kr-hier  kr-peri  kr-rand  pl-tree)
for graph in $graphs; do
    for size_param in data/$graph/*/; do
	size_param=$(basename $size_param)
	print "python build_sp_len.py ${graph} ${size_param}"
    done
done

graphs=(p2p-gnutella08 arxiv-hep-th enron-email)
for graph in $graphs; do
    print "python build_sp_len.py ${graph} \"\" "
done
    
