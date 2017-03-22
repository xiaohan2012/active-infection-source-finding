#! /bin/zsh
paths=$(cat /tmp/sp_len_paths)
# for p in $paths; do
if [[ -f /tmp/tranpose_cmds.txt ]]; then
    rm /tmp/tranpose_cmds.txt
fi
    
for p in `cat /tmp/sp_len_paths`; do
    # print $p
    print "zcat ${p} | split -l1 --numeric-suffixes --filter='gzip >> /tmp/\$FILE.gz'; rm ${p}" >> /tmp/tranpose_cmds.txt
done

cat /tmp/tranpose_cmds.txt | parallel -j8
