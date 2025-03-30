import csv
import yaml
from tqdm import tqdm

LANG_ISO3_TO_SCRIPT = dict()

with open("./glotscript.tsv", newline="") as tsv_f:
    reader = csv.reader(tsv_f, delimiter="\t")
    next(reader)
    for row in reader:
        LANG_ISO3_TO_SCRIPT[row[0]] = row[1]

LANG_ISO3_TO_ALL = dict()
LANG_ISO1_TO_ALL = dict()

with open("./languages_new.yml", "r") as yml_f:
    data_yml = yaml.safe_load(yml_f)

for row in data_yml:
    tmp_list = []
    lang_iso_3 = row[":iso_639_3"]
    lang_iso_1 = row[":iso_639_1"]
    lang_name = row[":name"].lower()
    if lang_iso_3 in LANG_ISO3_TO_SCRIPT.keys():
        lang_script = LANG_ISO3_TO_SCRIPT[lang_iso_3].split(", ")
    else:
        lang_script = None
    LANG_ISO3_TO_ALL[lang_iso_3] = [lang_iso_1, lang_name, lang_script]

    if lang_iso_1:
        LANG_ISO1_TO_ALL[lang_iso_1] = [lang_iso_3, lang_name, lang_script]
    else:
        continue

NLLB_TO_ALL = dict()
for key, val in LANG_ISO3_TO_ALL.items():
    if val[2]:
        for script in val[2]:
            nllb_code = f"{key}_{script}"
            lang_name = f"{val[1]} ({script})"
            NLLB_TO_ALL[nllb_code] = [key, val[0], val[1], script]
    else:
        continue