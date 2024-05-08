from glob import glob
import json
import os
from pathlib import Path

vctk_path = 'ttts/datasets/VCTK-Corpus'
paths = glob(os.path.join(vctk_path, "**/*.wav"), recursive=True)
out_path = 'ttts/datasets/vctk_data.jsonl'

for path in paths:
    path = Path(path)
    index = path.parts.index('wav48')
    text_path = (Path(*path.parts[:index])/Path('txt')/Path(*path.parts[index+1:]))
    text_path = Path(str(text_path).replace('.wav','.txt'))
    try:
        with open(text_path, "r", encoding='utf-8') as f:
            text = f.readline().replace('\n','')
        with open(out_path, 'a', encoding='utf-8') as file:
            json.dump({'text':text,'path':str(path)}, file, ensure_ascii=False)
            file.write('\n')
    except Exception as e:
        print(e)
        print(path)