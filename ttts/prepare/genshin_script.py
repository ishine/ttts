from glob import glob
import json
import os
from pathlib import Path

dataset_path = 'ttts/datasets/genshin'
paths = glob(os.path.join(dataset_path, "*/*/*.wav"))
out_path = 'ttts/datasets/genshin_data.jsonl'

for path in paths:
    text_path = Path(path)
    # index = path.parts.index('wav48')
    # text_path = (Path(*path.parts[:index])/Path('txt')/Path(*path.parts[index+1:]))
    text_path = Path(str(text_path).replace('.wav','.lab'))
    if "中文" in str(text_path):
        lang = "ZH"
    if "日语" in str(text_path):
        lang = "JP"
    if "英语" in str(text_path):
        lang = "EN"
    if "韩语" in str(text_path):
        lang = "KR"
    try:
        with open(text_path, "r", encoding='utf-8') as f:
            text = f.readline().replace('\n','')
        with open(out_path, 'a', encoding='utf-8') as file:
            json.dump({'text':text,'path':str(path),'lang':lang}, file, ensure_ascii=False)
            file.write('\n')
    except Exception as e:
        print(e)
        print(path)