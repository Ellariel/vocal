import os, sys
import numpy as np
import pandas as pd
from glob import glob
from scipy.stats import norm

from codes import CODES


def fleiss_kappa_with_p(df: pd.DataFrame, code):
    """
    Compute Fleiss' kappa, per-item agreement, z-score, and p-value.
    Assumes two categories: "for", "no decision".
    """
    # Encode categories
    matrix = df.map(lambda x: int(code.lower() in str(x).lower())).astype(int).to_numpy()
    #print(matrix)
    n_items, n_raters = matrix.shape

    # Counts per category per item
    counts = np.zeros((n_items, 2), dtype=int)
    counts[:, 0] = np.sum(matrix == 0, axis=1)
    counts[:, 1] = np.sum(matrix == 1, axis=1)

    # Per-item agreement
    P_i = (np.sum(counts**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    P_bar = np.mean(P_i)

    # Category proportions
    p_j = np.sum(counts, axis=0) / (n_items * n_raters)

    # Expected agreement
    P_e = np.sum(p_j**2)

    # Fleiss' kappa
    kappa = (P_bar - P_e) / (1 - P_e) if (1 - P_e) != 0 else np.nan

    # Variance of kappa (approximation)
    term1 = np.sum(p_j**2 * (1 - p_j)**2)
    term2 = (1 - P_e) * (np.sum(p_j**3) - P_e * np.sum(p_j**2))
    var_kappa = (term1 - term2) / (n_items * n_raters * (n_raters - 1) * (1 - P_e)**2)

    # z-score and p-value
    if var_kappa > 0:
        z = kappa / np.sqrt(var_kappa)
        p_value = 2 * (1 - norm.cdf(abs(z)))
    else:
        z, p_value = np.nan, np.nan

    return pd.Series(P_i, index=df.index, name="per_item_agreement"), kappa, z, p_value


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

    texts = [pd.read_excel(i, na_filter=False) for i in glob(os.path.join(texts_dir, "*.xls"))]
    texts = pd.concat(texts).reset_index(drop=True)
    print(f'N = {texts.shape[0]} (texts)')

    results = [pd.read_excel(i, na_filter=False) for i in glob(os.path.join(results_dir, "*.xlsx")) if '_code_' in i]
    results = pd.concat([r for r in results if len(r)]).reset_index(drop=True)
    print(f'n = {results.shape[0]} (model outputs)')

    results['result'] = results.apply(lambda x: x['tested_code'] if len(str(x['tested_code'])) and str(x['tested_code']).lower() in\
                                      str(x['code_applied']).lower()[:len(str(x['tested_code']))*2] else None, 
                                      axis=1)
    
    joined = pd.DataFrame()
    for idx, data in results.groupby('store_id'):
        item = data.iloc[:1].copy()
        for _, i in data.iterrows():
            item.loc[item.index, f"{i['tested_code']}.{i['model']}"] = i['result']
        item['confidence'] = pd.to_numeric(data['confidence'], errors='coerce').mean()
        joined = pd.concat([joined, item])

    #print(f'n = {joined.shape[0]}')
    joined = joined[[
        'store_id',
        'confidence',
    ] + [i for i in joined.columns 
                if i.startswith('code') and '_applied' not in i]]
    
    joined = pd.merge(joined, texts, left_on='store_id', right_on='StoreId', how='left')
    print(f'n = {joined.shape[0]} (results)')

    for code in codes:
        cols = [i for i in joined.columns if i.startswith(code + '.')]
        per_item_agreement, kappa, z, p_value = fleiss_kappa_with_p(joined[cols], code)
        joined[f"{code}_kappa_per_item"] = per_item_agreement
        joined[f"{code}_kappa_total"] = kappa
        joined[f"{code}_kappa_z_value"] = z
        joined[f"{code}_kappa_p_value"] = p_value
        joined[f"{code}_agreement_sum"] = joined.apply(lambda x: sum([int(code.lower() in str(x[i]).lower()) for i in cols]), axis=1)
        joined[f"{code}_agreement_rate"] = joined.apply(lambda x: sum([int(code.lower() in str(x[i]).lower()) for i in cols]) / len(cols), axis=1)

    joined.to_csv(os.path.join(results_dir, 'joined') + '.csv', sep=';', index=False)
    joined.to_excel(os.path.join(results_dir, 'joined') + '.xlsx', index=False)

    filtered = joined[joined.apply(lambda x: bool(sum([int(x[f"{code}_agreement_rate"] >= 0.5) for code in codes])), axis=1)]
    
    print(f'n = {filtered.shape[0]} (filtered)')
    filtered.to_csv(os.path.join(results_dir, 'filtered') + '.csv', sep=';', index=False)
    filtered.to_excel(os.path.join(results_dir, 'filtered') + '.xlsx', index=False)






   


