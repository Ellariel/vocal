import os, sys
import time
import random
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from openai import OpenAI

from codes import CODES
from prompt import base_prompt, parse_output



def run_with_new_instance(client, prompt, text, 
                          model="llama3.3:70b", 
                          temperature=1.5,
                          sleep=0,
                          seed=1313):
    
    random.seed(seed)
    np.random.seed(seed)
    reset_response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": "Forget about previous inquiries, instructions and prompts."}],
    )
    if sleep:
        time.sleep(sleep)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": f"Here's a piece of text, followed by instructions: “{text}”.\n{prompt}"
            },
        ],
        temperature=temperature,
        seed=seed
    )
    return completion.choices[0].message.content


def read_file(f):
    with open(f, 'r', encoding='utf-8') as file:
        return file.read()
    

def save_results(results, file_name):
        if 'confidence' in results:
            results['confidence'] = pd.to_numeric(results['confidence'], 
                                                  errors='coerce')
        cols = [i for i in [
            'model',
            'seed',
            'temperature',
            'version',
            'tested_code',        
            'code_applied',
            'confidence',
            'justification',
            'store_id',
            'output',
            'text_raw',
        ] if i in results.columns]
        results = results[cols]
        results.to_csv(file_name + '.csv', sep=';', index=False)
        results.to_excel(file_name + '.xlsx', index=False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, type=str)
    parser.add_argument('--ver', default='v1', type=str)
    parser.add_argument('--code', default=None, type=str)
    parser.add_argument('--model', default='llama3.3:70b', type=str)
    parser.add_argument('--temp', default=1.5, type=float)
    parser.add_argument('--seed', default=1313, type=int)
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__) if args.dir is None else args.dir
    texts_dir = os.path.join(base_dir, "texts")
    results_dir = os.path.join(base_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    codes = CODES[args.ver]

    print('model:', args.model)
    print('temp:', args.temp)
    print('seed:', args.seed)
    print('version:', args.ver)
    if args.code is not None:
        print('code:', args.code)
         

    texts = [pd.read_excel(i) for i in glob(os.path.join(texts_dir, "*.xls"))]
    texts = pd.concat(texts)

    print(f'N = {texts.shape[0]}')

    texts = texts[texts['language'].str.contains('English')]
    texts = texts[~texts['ArticleType'].str.contains('Scholarly Journals')].reset_index(drop=True) # .iloc[:5]

    n = texts.shape[0]
    print(f'n = {n}')

    texts['store_id'] = texts['StoreId']
    texts['text_raw'] = texts.apply(lambda x: f"{x['Title']}. {x['Abstract']}. {x['identifierKeywords']}.", axis=1)

    client = OpenAI(
        base_url = "https://open-webui.lcl.offis.de/api",
        api_key = read_file(os.path.join(base_dir, "apikey.key"))
    )


    def run_proccess(code, desc, texts):
        prompt = base_prompt(code,
                                desc['description'],
                                desc['keywords'])
            
        result_file = os.path.join(results_dir, 
                f"{args.model}_{args.seed}_{args.temp}_{args.ver}_{code}"\
                                .replace(':', '-'))
        
        if os.path.exists(f"{result_file}.csv"):
            results = pd.read_csv(f"{result_file}.csv", sep=';', na_filter=False,)
            n_done = results.shape[0]
        else:
            results = pd.DataFrame()
            n_done = 0
            
        if n_done < n:
            texts = texts.iloc[n_done:]
            for idx, item in tqdm(texts.iterrows(), total=len(texts), leave=False, desc=f"{code}/{args.model}"):
                    text = item['text_raw']
                    output = run_with_new_instance(client, prompt, text,
                                                temperature=args.temp,
                                                model=args.model,
                                                seed=args.seed)
                    # print(output)
                    r = parse_output(output)

                    r.update({'version' : args.ver,
                                'tested_code' : code,
                                'model' : args.model,
                                'temperature' : args.temp,
                                'seed' : args.seed,
                            })
                    r.update(item.to_dict())
                    results = pd.concat([results, pd.DataFrame([r])])
                    
                    if idx % 25 == 0:
                        save_results(results, result_file)

            save_results(results, result_file)
            print(f"code saved: {result_file}.csv")


    if args.code is not None:
        run_proccess(args.code, codes[args.code], texts)
    else:
        for code, desc in tqdm(codes.items(), leave=False):
            run_proccess(code, desc, texts)     