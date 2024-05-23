import json
from polyglot.detect import Detector
import cutlet
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from polyglot.detect.base import logger as polyglot_logger
from ttts.gpt.voice_tokenizer import VoiceBpeTokenizer
from tqdm import tqdm
import os

# 假设你的JSONL文件名为data.jsonl
filename = 'ttts/datasets/genshin_data.jsonl'

tok_zh = VoiceBpeTokenizer('ttts/tokenizers/zh_tokenizer.json')
tok_en = VoiceBpeTokenizer('ttts/tokenizers/en_tokenizer.json')
tok_jp = VoiceBpeTokenizer('ttts/tokenizers/jp_tokenizer.json')
tok_kr = VoiceBpeTokenizer('ttts/tokenizers/kr_tokenizer.json')
katsu = cutlet.Cutlet()
r = Transliter(academic)
# 用于筛选数据的判别函数
def is_eligible(data):
    text = data['text']
    lang = data['lang']
    if lang == "ZH":
        pass
        # text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        # text = ' '+text+' '
        # text = tok_zh.encode(text.lower())
    elif lang == "JP":
        pass
        # text = katsu.romaji(text)
        # text = ' '+text+' '
        # text = tok_jp.encode(text.lower())
    elif lang == "EN":
        pass
        # text = text
        # text = ' '+text+' '
        # text = tok_en.encode(text.lower())
    elif lang == "KR":
        pass
        # text = r.translit(text)
        # text = ' '+text+' '
        # text = tok_kr.encode(text.lower())
    else:
        print(text)
        return False, None
    path = data['path']
    sampling_rate = 48000
    size = os.path.getsize(path)
    duration = size / sampling_rate / 2
    if duration > 10 or duration < 1:
        print(text)
        return False, None
    return True, lang

# 用于存储符合条件的数据
ZH = []
JP = []
KR = []
EN = []

# 打开文件并逐行读取
with open(filename, 'r',encoding='utf-8') as file:
    for line in tqdm(file):
        # 解析每行的JSON数据
        data = json.loads(line)
        
        eli, lang = is_eligible(data)
        # 应用判别函数
        if eli:
            # 如果数据符合条件，则添加到列表中
            if lang == 'ZH':
                ZH.append(data)
            elif lang == 'JP':
                JP.append(data)
            elif lang == 'EN':
                EN.append(data)
            elif lang == 'KR':
                KR.append(data)
            else:
                pass
            # eligible_data.append(data)

# 现在eligible_data包含了所有符合条件的数据
# 你可以将其写入新的JSONL文件或进行其他操作
with open('ttts/datasets/genshin_data_zh.jsonl', 'w',encoding='utf-8') as file:
    for data in ZH:
        file.write(json.dumps(data, ensure_ascii=False) + '\n')
with open('ttts/datasets/genshin_data_en.jsonl', 'w',encoding='utf-8') as file:
    for data in EN:
        file.write(json.dumps(data, ensure_ascii=False) + '\n')
with open('ttts/datasets/genshin_data_kr.jsonl', 'w',encoding='utf-8') as file:
    for data in KR:
        file.write(json.dumps(data, ensure_ascii=False) + '\n')
with open('ttts/datasets/genshin_data_jp.jsonl', 'w',encoding='utf-8') as file:
    for data in JP:
        file.write(json.dumps(data, ensure_ascii=False) + '\n')
