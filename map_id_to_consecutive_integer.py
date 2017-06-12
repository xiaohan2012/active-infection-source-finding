from tqdm import tqdm
dataset = 'gplus'
inpath = 'data/{}/{}_combined.txt'.format(dataset, dataset)
outpath = 'data/{}/network.txt'.format(dataset)

ids = set()
total = 30494866

print('reading..')
with open(inpath, 'r') as f:
    for l in tqdm(f, total=total):
        i, j = map(int, l.split())
        ids.add(i)
        ids.add(j)

mapping = {n: i for i, n in enumerate(sorted(ids))}

print('writting..')
with open(outpath, 'w') as out_f:
    with open(inpath, 'r') as f:
        for l in tqdm(f, total=total):
            i, j = map(int, l.split())
            out_f.write("{} {}\n".format(mapping[i], mapping[j]))
