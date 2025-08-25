import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter, StrMethodFormatter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from glob import glob



def mark_dup(df):
    vectorizer = TfidfVectorizer().fit_transform(df.apply(lambda x: f"{x['authors']}{x['title']}{x['abstract']}", axis=1))
    cosine_sim = cosine_similarity(vectorizer)
    threshold = 0.8
    d = []
    for i in range(len(df)):
        for j in range(i + 1, len(df)):
            if cosine_sim[i, j] >= threshold:
                d.append((df.index[i], df.index[j], cosine_sim[i, j],))
    d = pd.DataFrame(d,
          columns=['i', 'j', 'dist'])
    df = pd.merge(df.reset_index(), d[['i']], how='left', left_on='index', right_on='i')
    df = pd.merge(df, d[['j', 'i']], how='left', left_on='index', right_on='j')
    df['cosine_id'] = df.apply(lambda x: x['i_x'] if pd.isna(x['i_y']) else x['i_y'], axis=1).astype('Int64')
    df.drop(['i_x', 'i_y', 'j'], axis=1, inplace=True)
    return df


if __name__ == "__main__":

    base_dir = os.path.dirname(__file__)
    texts_dir = os.path.join(base_dir, "texts")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(texts_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    rw = pd.read_excel(os.path.join(results_dir, 'data_final_checked.xlsx')).reset_index()
    rw = rw[['year', 'doi', 'authors', 'title', 'abstract', 'type', 'source']].copy()

    vl = pd.read_excel(os.path.join(results_dir, 'filtered.xlsx'))\
        .reset_index().rename(columns={'ArticleType': 'type'})
    vl.columns = [i.lower() for i in vl.columns]
    vl['source'] = vl.apply(lambda x: 'arxiv' if 'arxiv' in x['pubtitle'].lower() else 'proquest', axis=1)
    vl = vl[['year', 'authors', 'title', 'abstract', 'type', 'source']].copy()

    vl = vl.drop_duplicates(subset=['authors', 'title', 'abstract'])
    
    print(f"n = {vl.shape[0]}")

    df = pd.concat([vl, rw])

    d = mark_dup(df)

    d.to_csv(os.path.join(results_dir, 'vocal_dedup') + '.csv', sep=';', index=False)
    d.to_excel(os.path.join(results_dir, 'vocal_dedup') + '.xlsx', index=False)






   


