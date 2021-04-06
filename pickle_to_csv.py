import pandas as pd
import os

path = os.getcwd()

to_pickle = os.path.join(path, 'data', 'master.pkl')
to_csv = os.path.join(path, 'data', 'master.csv')

df = pd.read_pickle(to_pickle)

df.to_csv(to_csv, sep='\t')