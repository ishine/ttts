import json
import cutlet
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
from pypinyin import pinyin, lazy_pinyin, Style
from tqdm import tqdm
import os
import torchaudio

filename = ''
katsu = cutlet.Cutlet()
r = Transliter(academic)

def convert_to_latin(text, lang):
    if lang == "ZH":
        text = ' '.join(lazy_pinyin(text, style=Style.TONE3, neutral_tone_with_five=True))
        text = ' '+text+' '
    elif lang == "JP":
        text = katsu.romaji(text)
        text = ' '+text+' '
    elif lang == "EN":
        text = text
        text = ' '+text+' '
    elif lang == "KR":
        text = r.translit(text)
        text = ' '+text+' '
    else:
        return None
    return text

def process_jsonl(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
        for line in tqdm(infile):
            data = json.loads(line.strip())
            text = data['text']
            lang = data['lang']
            path = data['path']
            
            # 进行转换
            latin_text = convert_to_latin(text, lang)
            data['latin'] = latin_text

            wav,sr = torchaudio.load(path)
            data['wav_length'] = wav.shape[-1]
            data['sr'] = sr
            
            # 将处理后的数据写入输出文件
            json.dump(data, outfile, ensure_ascii=False)
            outfile.write('\n')

# 使用这个函数处理你的文件
input_file = 'ttts/datasets/genshin_data.jsonl'  # 输入文件名
output_file = 'ttts/datasets/genshin_data_latin.jsonl'  # 输出文件名
process_jsonl(input_file, output_file)