import os
import time
import argparse
from subprocess import Popen as new, CREATE_NEW_CONSOLE

from codes import CODES

MODELS = [
     #'llama3.1:8b',
     #'llama3.2:3b',
     #'phi4:14b',
     #'gemma3:27b',
     'llama3.3:70b',
     'deepseek-r1:70b',
     'gpt-oss:120b',
     'qwen3:235b',
     #'deepseek-r1:671b-0528',
]


def run_scenarios(temp, seed, ver, codes, base_dir, windows=False):
    """
    Run all.
    """

    threads = []
    params = dict() if not windows else dict(creationflags=CREATE_NEW_CONSOLE)

    for code in codes:
        for model in MODELS:
            print('<-----running scenario----->')
            threads.append(new(f"uv run {os.path.join(base_dir, 'exec.py')} --model {model} --temp {temp} --seed {seed} --ver {ver} --code {code}",
                                            **params))
            time.sleep(5)

    [i.wait() for i in threads]



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', default=None, type=str)
    parser.add_argument('--ver', default='v1', type=str)
    parser.add_argument('--code', default=None, type=str)
    parser.add_argument('--temp', default=1.5, type=float)
    parser.add_argument('--seed', default=1313, type=int)
    args = parser.parse_args()

    base_dir = os.path.dirname(__file__) if args.dir is None else args.dir
    codes = CODES[args.ver]

    print('temp:', args.temp)
    print('seed:', args.seed)
    print('version:', args.ver)
    if args.code is not None:
        print('code:', args.code)
    else:
        print('codes:', list(codes.keys()))

    if args.code is not None:
        run_scenarios(args.temp, args.seed, 
                      args.ver, [args.code], 
                      base_dir=base_dir)
    else:
        run_scenarios(args.temp, args.seed, 
                      args.ver, codes.keys(), 
                      base_dir=base_dir)



