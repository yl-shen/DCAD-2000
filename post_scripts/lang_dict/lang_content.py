"""
一些映射：目前有三个
+ LANG_ISO3_TO_SCRIPT
+ LANG_ISO3_TO_ALL
+ NLLB_TO_ALL
"""
import csv
import yaml
from tqdm import tqdm

# ***1. LANG ISO 639-3 到 script的映射 ---> glotscript
LANG_ISO3_TO_SCRIPT = dict()

with open("/mb-datacenter/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/code/dc_pipeline/post_scripts/lang_dict/glotscript.tsv", newline="") as tsv_f:
    reader = csv.reader(tsv_f, delimiter="\t")
    # 跳过表头
    next(reader)
    for row in reader:
        LANG_ISO3_TO_SCRIPT[row[0]] = row[1]
# 8035条记录 --- {'zho': 'Hanb, Arab, Brai, Bopo, Latn, Hant, Phag, Hans, Hani', 'zhw': 'Latn', 'zhx': 'Nshu',}
# print(len(LANG_ISO3_TO_SCRIPT))

# ***2. ISO 639-3 到其他的mapping -- 包括：iso_1, language_name, script（List类型）--- 这里的script 有可能是空/ iso_1也又可能是空
# ***4. ISO_1 to ALL --- 包括：iso_3, language_name, script（List类型）--- 这里的script 有可能是空/ iso_1也又可能是空
LANG_ISO3_TO_ALL = dict()
LANG_ISO1_TO_ALL = dict()

with open("/mb-datacenter/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/code/dc_pipeline/post_scripts/lang_dict/languages.yml", "r") as yml_f:
    # 列表类型，每个列表里边包含字典。注意有冒号
    data_yml = yaml.safe_load(yml_f)
# 遍历
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

# print(len(LANG_ISO3_TO_ALL))
# print(LANG_ISO1_TO_ALL)

# ***3. nllb_code to all  -- 包括: iso_3, iso_1, language_name, script (单个)， key是 nllb类型的code
NLLB_TO_ALL = dict()
for key, val in LANG_ISO3_TO_ALL.items():
    if val[2]:
        for script in val[2]:
            nllb_code = f"{key}_{script}"
            lang_name = f"{val[1]} ({script})"
            NLLB_TO_ALL[nllb_code] = [key, val[0], val[1], script]
    else:
        continue

# print(len(NLLB_TO_ALL))

## 2000多个code
# GLOT_LANG_CODE_LIST = []
# with open("/kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/code/dc_pipeline/post_scripts/lang_dict/glot_lang_code.txt", "r", encoding="utf-8") as f:
#     for line in f:
#         GLOT_LANG_CODE_LIST.append(line.strip())