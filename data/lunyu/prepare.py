import os
import requests
import tiktoken
import numpy as np
import json
import sentencepiece as spm

# download the tiny shakespeare dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input.json')
if not os.path.exists(input_file_path):
    data_url = 'https://raw.githubusercontent.com/hlthu/Chinese-Poetry-Dataset/master/lunyu/lunyu.json'
    data_url = 'https://raw.githubusercontent.com/hlthu/Chinese-Poetry-Dataset/master/json/poet.tang.0.json'
    with open(input_file_path, 'w') as f:
        f.write(requests.get(data_url).text)

with open(input_file_path, 'r') as f:
    js = json.load(f)

verses = []
for ch in js:
    verses += ch['paragraphs']
data = '\n'.join(verses)

with open(os.path.join(os.path.dirname(__file__), 'output.txt'), 'w') as f:
    f.write(data)

n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# train sentencepiece model from `botchan.txt` and makes `m.model` and `m.vocab`
# `m.vocab` is just a reference. not used in the segmentation.
spm.SentencePieceTrainer.train(f'--input=output.txt --model_prefix=m --vocab_size=1334')
# makes segmenter instance and loads the model file (m.model)
enc = spm.SentencePieceProcessor()
enc.load('m.model')

#enc = tiktoken.get_encoding("gpt2")
train_ids = enc.encode_as_ids(train_data)
val_ids = enc.encode_as_ids(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# train.bin has 301,966 tokens
# val.bin has 36,059 tokens
