import json
import cutlet
import torch
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from pypinyin import pinyin, lazy_pinyin, Style
from tqdm import tqdm
import os
import torchaudio
from process_romanize import romanize_file

def read_jsonl(path):
    with open(path, 'r') as f:
        json_str = f.read()
    data_list = []
    for line in json_str.splitlines():
        data = json.loads(line)
        data_list.append(data)
    return data_list
def romanize(file_paths, out_path, max_workers):
    paths = [[file_path, out_path] for file_path in file_paths]
    with torch.multiprocessing.get_context("spawn").Pool(max_workers) as pool:
        results = list(tqdm(pool.imap(romanize_file, paths), total=len(file_paths), desc="Roman"))
    results = [result for result in results if result is not None]

if __name__ == '__main__':
    # 使用这个函数处理你的文件
    # input_file = 'ttts/datasets/genshin_data.jsonl'  # 输入文件名
    # output_file = 'ttts/datasets/genshin_data_latin.jsonl'  # 输出文件名
    input_file = 'ttts/datasets/filtered_paths.jsonl'  # 输入文件名
    output_file = 'ttts/datasets/filtered_paths_latin.jsonl'  # 输出文件名
    jsons = read_jsonl(input_file)
    romanize(jsons, output_file,8)