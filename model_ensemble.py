import os
import random

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

result_dict = dict()
f_list = os.listdir('to_ensemble')

utt_num = 1425
pola_sum = np.zeros(1425)
itst_sum = np.zeros(1425)
emp_sum = np.zeros(1425)
pola_model_num = 0
itst_model_num = 0
emp_model_num = 0

for f in f_list:
    with open(f'to_ensemble/{f}', 'r') as f1:
        lines = f1.read().split('\n')[:-1]
        labels = np.array([float(line) for line in lines])
        if 'pola' in f:
            pola_sum += labels
            pola_model_num += 1
        elif 'itst' in f:
            itst_sum += labels
            itst_model_num += 1
        elif 'emp' in f:
            emp_sum += labels
            emp_model_num += 1

pola_esb = (pola_sum / pola_model_num).tolist()
itst_esb = (itst_sum / itst_model_num).tolist()
emp_esb = (emp_sum / emp_model_num).tolist()

print(f'Ensembel {pola_model_num} pola models, {itst_model_num} itst models and {emp_model_num} emp models. Outputs are as follows')
print(pola_esb)
print(itst_esb)
print(emp_esb)

df = pd.read_csv('wassa-test-final.csv')
df['pola_pred'] = pola_esb
df['itst_pred'] = itst_esb
df['emp_pred'] = emp_esb
df.to_csv('wassa_test_final_wi_pred.csv', encoding='gbk', index=False)

