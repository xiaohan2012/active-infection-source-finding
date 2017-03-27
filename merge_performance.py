import arrow
import pandas as pd


# paths = sys.argv[1:]
gtype = 'kr-hier'
utc = arrow.utcnow()
date = utc.to('Europe/Helsinki').format('YYYY-MM-DD')
            
output = 'data/{}/performance.pkl'.format(gtype)
paths = ['data/{}/performance-{}.pkl'.format(gtype, date), output]
dfs = [pd.read_pickle(path) for path in paths]
new_df = pd.concat(dfs)
new_df.to_pickle(output)
