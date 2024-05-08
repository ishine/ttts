import argparse
import functools
import multiprocessing
import os

import torch
import torchaudio
from tqdm import tqdm
from ttts.classifier.infer import read_jsonl
from ttts.prepare.extract_dur import process_dur
from ttts.utils.utils import find_audio_files, get_paths_with_cache

def process_wrapper(args):
   return process_dur(*args)

def extract_dur(file_paths, texts, max_workers):
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(process_wrapper, zip(file_paths, texts)), total=len(file_paths), desc="DUR"))
    # 过滤掉返回None的结果
    results = [result for result in results if result is not None]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path',default='ttts/datasets/44k_data.jsonl')
    args = parser.parse_args()
    paths = read_jsonl(args.json_path)
    texts = [path['text'] for path in paths]
    paths = [os.path.join('/home/hyc/tortoise_plus_zh',path['path']) for path in paths]
    num_threads = 10
    extract_dur(paths, texts, num_threads)
