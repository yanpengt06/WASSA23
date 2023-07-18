import numpy as np
import pandas as pd

df = pd.read_csv('temp.csv')
sigma_list = []
col_names = df.columns.tolist()[1:]

for idx, row in df.iterrows():
    att_plist = []
    for name in col_names:
        att_plist.append(row[name] * 100)
    sigma_list.append(np.var(att_plist))
df['variation'] = sigma_list
df.to_csv('temp.csv', index=False)