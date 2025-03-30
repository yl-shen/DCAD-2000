# Built from native speakers, with inspiration from
# https://github.com/zacanger/profane-words
# and
# https://github.com/thisandagain/washyourmouthoutwithsoap/blob/develop/data/build.json
# and
# https://github.com/LDNOOBW/List-of-Dirty-Naughty-Obscene-and-Otherwise-Bad-Words

import json
import os 
current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
current_file_dir = os.path.dirname(current_file_dir)

JSON_PATH = f"{current_file_dir}/stop_words/toxity-200.json"

with open(JSON_PATH, 'r', encoding='utf-8') as json_f:
    flagged_words = json.load(json_f)
