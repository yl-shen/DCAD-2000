# Built from native speakers, with inspiration from
# https://github.com/6/stopwords-json
# and
# https://github.com/stopwords-iso/stopwords-iso for Urdu and Vietnamese

import json
import os
import yaml


### Stop Words from fineweb-2
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
current_file_dir = os.path.dirname(current_file_dir)

CONFIG_PATH = f"{current_file_dir}/stop_words/stopwords_config"
stopwords = dict()

file_list = os.listdir(CONFIG_PATH)

for file in file_list:
    lang = file.split(".")[0]
    with open(os.path.join(CONFIG_PATH, file), "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
        stopwords[lang] = data["stopwords"]