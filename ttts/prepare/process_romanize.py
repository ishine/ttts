import json
import cutlet
from hangul_romanize import Transliter
from hangul_romanize.rule import academic
import torchaudio
from pypinyin import pinyin, lazy_pinyin, Style

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

def romanize_file(paths):
    in_path, out_path = paths

    data = in_path
    text = data['text']
    path = data['path']
    try:
        lang = data['lang']
    except:
        lang="ZH"
    
    # 进行转换
    latin_text = convert_to_latin(text, lang)
    data['latin'] = latin_text

    wav,sr = torchaudio.load(path)
    data['wav_length'] = wav.shape[-1]
    data['sr'] = sr
    
    # 将处理后的数据写入输出文件
    with open(out_path, 'a', encoding='utf-8') as outfile:
        json.dump(data, outfile, ensure_ascii=False)
        outfile.write('\n')