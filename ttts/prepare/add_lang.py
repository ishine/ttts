import json
from tqdm import tqdm
filename = 'ttts/datasets/44k_data.jsonl'
OUT = []

# 打开文件并逐行读取
with open(filename, 'r',encoding='utf-8') as file:
    for line in tqdm(file):
        # 解析每行的JSON数据
        data = json.loads(line)
        data['lang'] = 'ZH'
        
        OUT.append(data)

with open('ttts/datasets/44k_data_zh.jsonl', 'w',encoding='utf-8') as file:
    for data in OUT:
        file.write(json.dumps(data, ensure_ascii=False) + '\n')
