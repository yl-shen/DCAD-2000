import os
import pandas as pd

### NLLB in CLUTURAX
CULTURAX_To_NLLB = {
    'as': 'asm_Beng',
    'su': 'sun_Latn',
    'mk': 'mkd_Cyrl',
    'hr': 'hrv_Latn',
    'tr': 'tur_Latn',
    'yo': 'yor_Latn',
    'he': 'heb_Hebr',
    'pt': 'por_Latn',
    'ast': 'ast_Latn',
    'hy': 'hye_Armn',
    'vi': 'vie_Latn',
    'gl': 'glg_Latn',
    'ckb': 'ckb_Arab',
    'fr': 'fra_Latn',
    'vec': 'vec_Latn',
    'ka': 'kat_Geor',
    'eu': 'eus_Latn',
    'nl': 'nld_Latn',
    'hu': 'hun_Latn',
    'si': 'sin_Sinh',
    'bg': 'bul_Cyrl',
    'ur': 'urd_Arab',
    'es': 'spa_Latn',
    'el': 'ell_Grek',
    'my': 'mya_Mymr',
    'oc': 'oci_Latn',
    'da': 'dan_Latn',
    'lo': 'lao_Laoo',
    'eo': 'epo_Latn',
    'sd': 'snd_Arab',
    'th': 'tha_Thai',
    'kn': 'kan_Knda',
    'km': 'khm_Khmr',
    'af': 'afr_Latn',
    'ht': 'hat_Latn',
    'pl': 'pol_Latn',
    'lb': 'ltz_Latn',
    'id': 'ind_Latn',
    'ja': 'jpn_Jpan',
    'de': 'deu_Latn',
    'cy': 'cym_Latn',
    'hi': 'hin_Deva',
    'lt': 'lit_Latn',
    'so': 'som_Latn',
    'sa': 'san_Deva',
    'gn': 'grn_Latn',
    'sv': 'swe_Latn',
    'tg': 'tgk_Cyrl',
    'be': 'bel_Cyrl',
    'gd': 'gla_Latn',
    'sr': 'srp_Cyrl',
    'tt': 'tat_Cyrl',
    'sl': 'slv_Latn',
    'en': 'eng_Latn',
    'scn': 'scn_Latn',
    'ug': 'uig_Arab',
    'tk': 'tuk_Latn',
    'li': 'lim_Latn',
    'ml': 'mal_Mlym',
    'te': 'tel_Telu',
    'ky': 'kir_Cyrl',
    'azb': 'azb_Arab',
    'cs': 'ces_Latn',
    'sk': 'slk_Latn',
    'nn': 'nno_Latn',
    'fi': 'fin_Latn',
    'ceb': 'ceb_Latn',
    'ga': 'gle_Latn',
    'ca': 'cat_Latn',
    'bs': 'bos_Latn',
    'it': 'ita_Latn',
    'am': 'amh_Ethi',
    'et': 'est_Latn',
    'kk': 'kaz_Cyrl',
    'mai': 'mai_Deva',
    'uk': 'ukr_Cyrl',
    'ro': 'ron_Latn',
    'is': 'isl_Latn',
    'ko': 'kor_Hang',
    'jv': 'jav_Latn',
    'war': 'war_Latn',
    'gu': 'guj_Gujr',
    'ba': 'bak_Cyrl',
    'sw': 'swh_Latn',
    'arz': 'arz_Arab',
    'mr': 'mar_Deva',
    'ne': 'npi_Deva',
    'mt': 'mlt_Latn',
    'ta': 'tam_Taml',
    'ru': 'rus_Cyrl',
    'or': 'ory_Orya',
    'lmo': 'lmo_Latn',
    # after check (another 18 languages)
    'fa': 'pes_Arab',
    'mn': 'khk_Cyrl',
    'zh': 'zho_Hans',
    'uz': 'uzn_Latn',
    'sq': 'als_Latn',
    'ps': 'pbt_Arab',
    'tt': 'crh_Latn',
    'ar': 'arb_Arab',
    'qu': 'quy_Latn',
    'yi': 'ydd_Hebr',
    'mg': 'plt_Latn',
    'lv': 'lvs_Latn',
    'min': 'min_Latn',
    'ku': 'kmr_Latn',
    'no': 'nob_Latn',
    'bo': 'bod_Tibt',
    'ms': 'zsm_Latn',
    'yue': 'yue_Hant',
}

## Maaping
# BCP_TO_ISO = {}
# ISO_TO_BCP = {}
# ISO_TO_GLOT = {}

# mapping_file = '/home/hadoop/zhangxueren/jupyter_zhangxueren/shenyingli/code/dc_pipeline/lang_mapping/google_lang_mapping.tsv'
# mapping_pd = pd.read_csv(mapping_file, sep="\t")
# ## headers: ['ISO 639 code', 'BCP-47 code', 'Number of speakers (rounded)', 'Writing system(s), ISO 15924', 'Name (Glottolog)', 'Glottocode (Glottolog)', 'Region, ISO 3166 (Glottolog)', 'Link to Glottolog', 'Alternative Name(s)', 'Endonym', 'Link to Wikipedia']

# for idx, row in mapping_pd.iterrows():
#     # BCP_TO_ISO[row['BCP-47 code']] = row['ISO 639 code']
#     # ISO_TO_BCP[row["ISO 639 code"]] = row["BCP-47 code"]
#     ISO_TO_GLOT[row["ISO 639 code"]] = [str(row["ISO 639 code"]) + '-' + script for script in row["Writing system(s), ISO 15924"].split(',')]

# print(ISO_TO_GLOT)

# import csv

# BCP_TO_ISO = {}
# bcp_iso_file = "/home/hadoop/zhangxueren/jupyter_zhangxueren/shenyingli/code/dc_pipeline/lang_mapping/bcp_to_iso.csv"
# with open(bcp_iso_file, 'r', encoding='utf-8-sig') as iso_file:
#     for line in iso_file:
#         tmp_list = line.strip().split(";")
#         BCP_TO_ISO[tmp_list[0]] = tmp_list[1]

# mapping_file = '/home/hadoop/zhangxueren/jupyter_zhangxueren/shenyingli/code/dc_pipeline/lang_mapping/all_mapping.csv'
# bcp_file = '/home/hadoop/zhangxueren/jupyter_zhangxueren/shenyingli/code/dc_pipeline/lang_mapping/BCP47.csv'
# mapping_pd = pd.read_csv(mapping_file, sep=";")
# bcp47_pd = pd.read_csv(bcp_file, sep=",")

# total_list = []
# for idx, row in bcp47_pd.iterrows():
#     tmp_list = []
#     tmp_list.append(row['lang'])
#     tmp_list.append(row['name'])
#     tmp_list.append(row['script'])
#     ## 添加脚本表示
#     #1. 先根据语言名字获取 ISO-3
#     iso_639_3 = mapping_pd.loc[mapping_pd['English_Name'] == row['name'], '639-3']
#     if iso_639_3.empty:
#         tmp_list.append(f"{BCP_TO_ISO[row['lang']]}_{row['script']}")
#     else:
#         tmp_list.append(f"{iso_639_3.values[0]}_{row['script']}")
#     total_list.append(tmp_list)

# with open("/home/hadoop/zhangxueren/jupyter_zhangxueren/shenyingli/code/dc_pipeline/lang_mapping/madlad.csv", mode="w", newline="", encoding="utf-8") as mad_file:
#     writer = csv.writer(mad_file)
#     writer.writerows(total_list)


# NLLB_LID_LABEL_LIST = []
# import fasttext
# lid_path = "/home/hadoop/zhangxueren/jupyter_zhangxueren/shenyingli/hf_model/fasttext-language-identification/nllb_200/model.bin"
# LID_MODEL = fasttext.load_model(lid_path)
# all_supported_list = LID_MODEL.labels
# NLLB_LID_LABEL_LIST = [lang.replace('__label__', "") for lang in all_supported_list]
# print(NLLB_LID_LABEL_LIST)

# GLOTLID_LID_LABEL_LIST = []
# import fasttext
# lid_path = "/home/hadoop/zhangxueren/jupyter_zhangxueren/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin"
# LID_MODEL = fasttext.load_model(lid_path)
# all_supported_list = LID_MODEL.labels
# GLOTLID_LID_LABEL_LIST = [lang.replace('__label__', "") for lang in all_supported_list]
# print(GLOTLID_LID_LABEL_LIST)

SP_LM_LIST = []
sp_path = "/data/shenyingli/hf_model/sp_lm"
# sp_path = "/kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/sp_lm"
file_list = os.listdir(sp_path)
for file in file_list:
    tmp_list = file.split(".")
    if tmp_list[0] not in SP_LM_LIST:
        SP_LM_LIST.append(tmp_list[0])

# print(SP_LM_LIST)
