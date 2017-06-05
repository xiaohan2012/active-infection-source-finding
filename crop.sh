#! /bin/zsh


dirs=("barabasi-64_by_models"      
      "p2p-gnutella08_by_models"
      "by_datasets"
      "illustration")
dirs=("illustration")

for dir in ${dirs}; do
    for path in figs/paper_experiment/${dir}/*.pdf; do
	print "pdfcrop ${path} ${path}"
    done
done

