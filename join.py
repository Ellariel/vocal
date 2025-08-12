import os, sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob

from codes import CODES

def get_item(i, l):
    if len(l):
        try:
            return l[i]
        except:
            pass


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    texts_dir = os.path.join(base_dir, "texts")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(texts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    codes = CODES['v1'].keys()
    results = [pd.read_excel(i) for i in glob(os.path.join(results_dir, "*.xlsx")) if 'joined' not in i]
    results = pd.concat(results).reset_index(drop=True)
    print(f'N = {results.shape[0]}')

    results['result'] = results.apply(lambda x: x['tested_code'] if x['tested_code'].lower() in\
                                        str(x['code_applied']).lower()[:len(x['tested_code'])*2] else None, 
                                      axis=1)
    
    joined = pd.DataFrame()
    for idx, data in results.groupby('store_id'):
        item = data.iloc[:1].copy()
        for _, i in data.iterrows():
            item.loc[item.index, f"{i['tested_code']}_{i['model']}"] = i['result']
        item['confidence'] = data['confidence'].mean()
        joined = pd.concat([joined, item])

    print(f'n = {joined.shape[0]}')
    joined = joined[[
        'store_id',
        'confidence',
    ] + [i for i in joined.columns 
                if i.startswith('code') and '_applied' not in i]]

    joined.to_csv(os.path.join(results_dir, 'joined') + '.csv', sep=';', index=False)
    joined.to_excel(os.path.join(results_dir, 'joined') + '.xlsx', index=False)





   


