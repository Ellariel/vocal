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

    codes = list(CODES['v1'].keys())

    texts = [pd.read_excel(i) for i in glob(os.path.join(texts_dir, "*.xls"))]
    texts = pd.concat(texts).reset_index(drop=True)
    print(f'N = {texts.shape[0]} (texts)')

    results = [pd.read_excel(i) for i in glob(os.path.join(results_dir, "*.xlsx")) if 'joined' not in i]
    results = pd.concat(results).reset_index(drop=True)
    print(f'n = {results.shape[0]} (model outputs)')

    results['result'] = results.apply(lambda x: x['tested_code'] if len(str(x['tested_code'])) and str(x['tested_code']).lower() in\
                                      str(x['code_applied']).lower()[:len(str(x['tested_code']))*2] else None, 
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
    
    joined = pd.merge(joined, texts, left_on='store_id', right_on='StoreId', how='left')
    print(f'n = {joined.shape[0]} (results)')

    for code in codes:
        cols = [i for i in joined.columns if i.startswith(code + '_')]
        joined[code] = joined.apply(lambda x: 1 if sum([int(pd.notna(x[i])) for i in cols]) / len(cols) >= 0.5 else 0, axis=1)

    joined.to_csv(os.path.join(results_dir, 'joined') + '.csv', sep=';', index=False)
    joined.to_excel(os.path.join(results_dir, 'joined') + '.xlsx', index=False)

    filtered = joined[joined.apply(lambda x: bool(sum([x[code] for code in codes])), axis=1)]
    
    print(f'n = {filtered.shape[0]} (filtered)')
    filtered.to_csv(os.path.join(results_dir, 'filtered') + '.csv', sep=';', index=False)
    filtered.to_excel(os.path.join(results_dir, 'filtered') + '.xlsx', index=False)






   


