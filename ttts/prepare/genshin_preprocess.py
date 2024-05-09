import re
from pypinyin import pinyin, lazy_pinyin, Style
import os
import glob
import cutlet
katsu = cutlet.Cutlet()

# 设置包含子文件夹和.lab文件的目录路径
base_dir = 'ttts/datasets/韩语 - Korean'

# 用于存储所有.lab文件内容的列表
all_lab_contents = []

# 使用glob模块找到所有.lab文件
for lab_file in glob.glob(os.path.join(base_dir, '**', '*.lab'), recursive=True):
    with open(lab_file, 'r', encoding='utf-8') as file:
        # 读取文件内容并添加到列表中
        all_lab_contents.append(file.read())

# 将所有内容写入一个新的文件中
with open('ttts/data/bpe_train-set.txt', 'w', encoding='utf-8') as combined_file:
    for content in all_lab_contents:
        # content = katsu.romaji(content)
        # content = ' '.join(lazy_pinyin(content, style=Style.TONE3, neutral_tone_with_five=True))
        combined_file.write(content + '\n')  # 每个文件内容后添加换行符
