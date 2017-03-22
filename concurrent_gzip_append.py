import gzip
import fcntl
from joblib import Parallel, delayed
from tqdm import tqdm

M = 4096
N = 4096
D = 100
s = '11 '


def write_matrices(path, M, N):
    with gzip.GzipFile(path, 'wb') as f:
        for i in range(M):
            f.write((s * N + '\n').encode('utf-8'))

input_paths = ['/tmp/mn-{}'.format(i)
               for i in range(D)]
Parallel(n_jobs=-1)(delayed(write_matrices)(p, M, N)
                    for p in tqdm(input_paths))


def append_to_file(input_path, paths):
    with gzip.GzipFile(input_path, 'rb') as f:
        for p, l in zip(paths, f):
            with gzip.GzipFile(p, 'ab') as ofile:
                fcntl.flock(ofile, fcntl.LOCK_EX)
                ofile.write(l)
                fcntl.flock(ofile, fcntl.LOCK_UN)

output_paths = ['/tmp/nd-{}'.format(i) for i in range(M)]
Parallel(n_jobs=-1)(delayed(append_to_file)(p, output_paths)
                    for p in tqdm(input_paths))

for p in tqdm(output_paths):
    with gzip.GzipFile(p, 'rb') as f:
        r = f.read()
