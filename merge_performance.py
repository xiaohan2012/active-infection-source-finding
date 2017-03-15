import sys
import pandas as pd


# paths = sys.argv[1:]
output = 'data/pl-tree/performance.pkl'
paths = ['data/pl-tree/performance-2017-03-15.pkl', output]
dfs = [pd.read_pickle(path) for path in paths]
new_df = pd.concat(dfs)
new_df.to_pickle(output)
