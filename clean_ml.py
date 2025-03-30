import argparse, os
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import time

from filtering import LoadParameters, ModifyingDocuments, Filtering, FunctionDatasetModifyingDocuments

## 启动sklearn 加速
# from sklearnex import patch_sklearn
# patch_sklearn() ## 启动加速补丁

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

import fasttext
# import sentencepiece
# import kenlm

"""
需要更改的参数：现在支持的/
--- process_type: all/one  --- all:指定语言下的所有文件，适合在集群里跑的时候指定。one:该语言下选择一个文本去处理，适合测试的时候使用
--- data_type: fineweb/fineweb_removed/mala/nllb             /oscar/aya /hplt/glotcc -- 暂时只支持fineweb/cultuax/madlad，后面再加
--- method_type: kmeans / oc_svm / iso_forest /lof
--- lang_list: iso-639-3加上下划线加上语言script版本，多个语言的话，需要空格分开
--- draw_fig: 如果需要生成图片就加上 --draw_fig， 如果在集群里就不要加

python clean_ml.py \
--draw_fig \
--process_type one \
--lang_list ara_Arab \
--num_proc 28 \
--data_type mala \
--method_type iso_forest \
--cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/.cache \
--data_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/huggingface/mala-monolingual-filter \
--lid_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin \
--lm_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_lm \
--sp_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_sp \
--out_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out/test_clean_ml


python clean_ml.py \
--draw_fig \
--process_type one \
--lang_list zho_Hans \
--num_proc 28 \
--data_type nllb \
--method_type kmeans \
--cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/.cache \
--data_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/datasets/mono/nllb_parquet \
--lid_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin \
--lm_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_lm \
--sp_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_sp \
--out_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out/test_clean_ml

--------------------


python clean_ml.py \
--process_type all \
--lang_list aaa_Latn abc_Latn abk_Cyrl abq_Cyrl abt_Latn ace_Latn acf_Latn ach_Latn acm_Arab acr_Latn ada_Latn adh_Latn ady_Cyrl ady_Latn aeu_Latn afb_Arab afr_Latn agq_Latn agr_Latn ags_Latn ahk_Latn aia_Latn ajp_Arab ajz_Latn aka_Latn akb_Latn aln_Latn als_Latn alt_Cyrl alz_Latn ame_Latn amh_Ethi ami_Latn amp_Latn amu_Latn ang_Latn ang_Runr ann_Latn anp_Deva anp_Latn aoj_Latn apc_Arab ape_Latn aph_Deva ara_Arab arb_Arab arc_Latn arc_Syrc arg_Latn arn_Latn ary_Arab arz_Arab asm_Beng ast_Latn atj_Latn ava_Cyrl ava_Latn avk_Cyrl avk_Latn awa_Deva awa_Latn awb_Latn aym_Latn azb_Arab aze_Arab aze_Latn azj_Latn azn_Latn azo_Latn bag_Latn bak_Cyrl bak_Latn bal_Arab bam_Latn ban_Latn bar_Latn bas_Latn bat_Cyrl bat_Latn baw_Latn bax_Latn bbc_Latn bbj_Latn bbk_Latn bcc_Arab bce_Latn bci_Latn bcl_Latn bef_Latn bel_Cyrl bem_Latn ben_Beng ben_Latn ber_Latn bew_Cyrl bfd_Latn bfm_Latn bfn_Latn bgf_Latn bgp_Latn bho_Deva bhs_Latn bih_Deva bik_Latn bim_Latn bis_Latn bjn_Latn bjr_Latn bkc_Latn bkh_Latn bkm_Latn bkx_Latn blk_Latn blk_Mymr bob_Latn bod_Tibt bos_Latn boz_Latn bpy_Beng bqc_Latn bqm_Latn bra_Deva brb_Khmr bre_Latn bri_Latn bru_Latn brv_Laoo brx_Deva bss_Latn bts_Latn btx_Latn bua_Cyrl bud_Latn bug_Bugi bug_Latn bul_Cyrl bum_Latn buo_Latn bus_Latn bwt_Latn bwx_Latn bxa_Latn bxr_Cyrl bxr_Latn bya_Latn bze_Latn bzi_Thai bzj_Latn cab_Latn cac_Latn cak_Latn cat_Latn cbk_Latn cbr_Latn cce_Latn cdo_Hani cdo_Latn ceb_Latn ces_Latn cfm_Latn cgc_Latn cha_Latn chd_Latn che_Cyrl chk_Latn chm_Cyrl chp_Cans chr_Cher chr_Latn chu_Cyrl chu_Latn chv_Cyrl chy_Latn cim_Latn cjs_Cyrl cjs_Latn ckb_Arab ckt_Cyrl ckt_Latn clo_Latn cmo_Khmr cnh_Latn cni_Latn cor_Latn cos_Latn cre_Cans cre_Latn crh_Cyrl crh_Latn crs_Latn csb_Latn csw_Cans csw_Latn csy_Latn ctd_Latn ctu_Latn cuh_Latn cuk_Latn cuv_Latn cym_Latn dag_Arab dag_Latn dan_Latn dar_Cyrl ddg_Latn ded_Latn deu_Latn dig_Latn din_Latn diq_Latn div_Thaa dje_Latn djk_Latn dln_Latn dmg_Latn dnw_Latn doi_Deva dov_Latn dsb_Latn dtp_Latn dtr_Latn dty_Deva dty_Latn dug_Latn dwr_Ethi dwr_Latn dyu_Latn dzo_Tibt eee_Thai ekk_Latn ekm_Latn ell_Grek eml_Latn emp_Cyrl emp_Latn enb_Latn enc_Latn eng_Latn enq_Latn epo_Latn est_Latn eus_Latn eve_Cyrl evn_Cyrl ewe_Latn ewo_Latn ext_Latn fao_Latn fas_Arab fat_Latn ffm_Latn fij_Latn fil_Latn fin_Latn fip_Latn fiu_Cyrl fiu_Latn fli_Latn fon_Latn fra_Latn frp_Latn frr_Latn fry_Latn fub_Latn fuh_Latn ful_Latn fur_Latn gag_Cyrl gag_Latn gal_Latn gan_Hani gan_Latn gbj_Orya gbm_Deva gcr_Latn gla_Latn gld_Cyrl gle_Latn glg_Latn glk_Arab glv_Latn gof_Latn gom_Deva gom_Latn gor_Latn got_Goth got_Latn gou_Latn gpe_Latn grc_Grek grn_Latn gsw_Latn gub_Latn guc_Latn gug_Latn guh_Latn gui_Latn guj_Deva guj_Gujr gur_Latn guw_Latn gvl_Latn gwc_Arab gym_Latn \
--num_proc 28 \
--data_type mala \
--method_type iso_forest \
--cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/.cache \
--data_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/huggingface/mala-monolingual-filter \
--lid_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin \
--lm_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_lm \
--sp_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_sp \
--out_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out_colony


python clean_ml.py \
--process_type all \
--lang_list kan_Knda kat_Geor kaz_Cyrl khm_Khmr kir_Cyrl kmr_Latn kom_Cyrl kor_Hang krc_Cyrl \
--num_proc 28 \
--data_type oscar \
--oscar_version 2024-30 \
--method_type iso_forest \
--cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/.cache/oscar \
--data_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/datasets/mono/oscar-corpus \
--lid_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin \
--lm_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_lm \
--sp_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_sp \
--out_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out_colony


### 以下是指定一个文件名取获取pdf的例子
python clean_ml.py \
--draw_fig \
--process_type one_specific \
--one_specific_file ** \
--lang_list eng_Latn \
--num_proc 28 \
--data_type oscar \
--oscar_version 2024-30 \
--method_type iso_forest \
--cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/.cache/oscar \
--data_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/datasets/mono/oscar-corpus \
--lid_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin \
--lm_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_lm \
--sp_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_sp \
--out_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/**


"""

current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)

## 设置参数
parser = argparse.ArgumentParser(description="select the filter parameters dynamiclly.")
parser.add_argument("--lang_list", type=str, nargs='+', help='language list to clean')
parser.add_argument("--character_repetition_length", type=int, default=10)
parser.add_argument("--draw_fig", action="store_true", help="draw the statistic figure?")
parser.add_argument("--num_proc", type=int, default=16)
parser.add_argument("--data_type", type=str, default="")
parser.add_argument("--process_type", type=str, default="")
parser.add_argument("--method_type", type=str, default="")
parser.add_argument("--cache_dir", type=str, default="")
parser.add_argument("--data_path", type=str, default="")
parser.add_argument("--out_path", type=str, default="")
parser.add_argument("--lid_path", type=str, default="")
parser.add_argument("--sp_path", type=str, default="")
parser.add_argument("--lm_path", type=str, default="")
parser.add_argument("--oscar_version", type=str, default="2024-30")
parser.add_argument("--one_specific_file", type=str, default="", help="指定一个文件，注意只传入文件名即可")

args = parser.parse_args()

# LID_PATH = os.path.join(current_file_dir, '..', '..', 'hf_model', 'fasttext-language-identification', 'glotlid', 'model_v3.bin')
## GLOT CONFIG PATH
GLOT_CONFIG_PATH = os.path.join(current_file_dir, "post_scripts", "lang_dict", "glot_mapping.csv")

GLOT_TO_ISO_1 = dict()
with open(GLOT_CONFIG_PATH, "r", encoding="utf-8") as f:
    for row in f:
        tmp_list = row.strip().split(",")
        GLOT_TO_ISO_1[tmp_list[0]] = tmp_list[2]

## load glot lid model
# lid_model  = fasttext.load_model(LID_PATH)
lid_model = fasttext.load_model(args.lid_path)

def draw_figure(stats_columns, stats_data, clus, args, lang, file_name):
    # Create a 4x2 grid of subplots
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    axes = axes.flatten()  # Flatten the axes array for easy iteration

    # Plot each statistic
    for i, stat in enumerate(stats_columns):
        ax = axes[i]
        ax.scatter(range(len(stats_data)), [row[i] for row in stats_data], c=clus, cmap="viridis", alpha=0.6)
        ax.set_title(stat)
        ax.set_xlabel("Sample Index")
        ax.set_ylabel("Value")
        ax.grid(True)
    
    # Remove unused subplots (if any)
    for j in range(len(stats_columns), len(axes)):
        fig.delaxes(axes[j])

    # Adjust layout and show plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.suptitle("Clustering/Detecting Results: High vs Low Quality Data", fontsize=16)
    if args.data_type == "oscar":
        plt.savefig(f"{args.out_path}/{args.data_type}/{args.oscar_version}/{lang}/{args.data_type}_{args.method_type}_{lang}_{os.path.splitext(file_name)[0]}.pdf")
    else:
        plt.savefig(f"{args.out_path}/{args.data_type}/{lang}/{args.data_type}_{args.method_type}_{lang}_{os.path.splitext(file_name)[0]}.pdf")


## 有效的后缀
VALID_SUFFIX = ["json", "parquet", "arrow", "txt"]

def main(args):
    lang_list = args.lang_list
    for lang in lang_list:
        ## 确保目录存在
        if args.data_type == "oscar":
            os.makedirs(os.path.join(args.out_path, args.data_type, args.oscar_version, lang), exist_ok=True)
        else:
            os.makedirs(os.path.join(args.out_path, args.data_type, lang), exist_ok=True)

        param = LoadParameters.load_parameters(lang)
        stopwords = LoadParameters.load_stopwords(lang)
        flagged_words = LoadParameters.load_flagged_words(lang)
        # lid_model = LoadParameters.load_model_lang_id(LID_PATH)
        if lang in GLOT_TO_ISO_1.keys() and os.path.exists(f"{args.sp_path}/{GLOT_TO_ISO_1[lang]}.sp.model"):
            try:
                sp_model = LoadParameters.load_sentencepiece_model(f"{args.sp_path}/{GLOT_TO_ISO_1[lang]}.sp.model")
                lm_model = LoadParameters.load_kenlm_model(f"{args.lm_path}/{GLOT_TO_ISO_1[lang]}.arpa.bin")
            except:
                sp_model = None
                lm_model = None
        else:
            sp_model = None
            lm_model = None
        

        def statistic_mapping(example):
            # def statistic_mapping(example):
            text = example["text"]
            ## 1. number_words
            words_list = ModifyingDocuments.get_words_from_document(text, sp_model, lower_case=False, strip_characters=param["strip_characters"])
            example["num_words"] = len(words_list)
            ## 2. character_repetition_ratio
            character_repetition_ratio = Filtering.compute_character_repetition_ratio(text, 10)
            example["character_repetition_ratio"] = round(character_repetition_ratio, 3)
            ## 3. word_repetition_ratio
            word_repetition_ratio = Filtering.compute_word_repetition_ratio(text, sp_model, param["strip_characters"], 10)
            example["word_repetition_ratio"] = round(word_repetition_ratio, 3)
            ## 4. special_characters_ratio
            special_characters_ratio = Filtering.compute_special_characters_ratio(text, param["special_characters"])
            example["special_characters_ratio"] = round(special_characters_ratio, 3)
            ## 5. stopwords_ratio
            if stopwords:
                stopwords_ratio = Filtering.compute_stopwords_ratio(text, sp_model, param["strip_characters"], param["cond_words_augmentation"], param["words_augmentation_group_sizes"], param["words_augmentation_join_char"], stopwords)
                example["stopwords_ratio"] = round(stopwords_ratio, 3)
            else:
                example["stopwords_ratio"] = 0.0
            ## 6. flagged_words_ratio
            if flagged_words:
                flagged_words_ratio = Filtering.compute_flagged_words_ratio(text, sp_model, param["strip_characters"], param["cond_words_augmentation"], param["words_augmentation_group_sizes"], param["words_augmentation_join_char"], flagged_words)
                example["flagged_words_ratio"] = round(flagged_words_ratio, 3)
            else:
                example["flagged_words_ratio"] = 0.0
            ## 7. lang_id_score
            document = text.lower().replace("\n", " ")
            pred = lid_model.predict(document)
            score_pred = pred[1][0]
            score_pred = round(score_pred, 3)
            example["lang_id_score"] = score_pred
            ## 8. perplexity_score
            if lm_model:
                perplexity_score = Filtering.compute_perplexity_score(text, sp_model, lm_model)
                example["perplexity_score"] = round(perplexity_score, 3)
            else:
                example["perplexity_score"] = 500.0
            return example

        ## file_list
        if args.data_type in ["culturax", "madlad"]:
            file_list = os.listdir(os.path.join(args.data_path, GLOT_TO_ISO_1[lang]))
        elif args.data_type == "fineweb":
            file_list = os.listdir(os.path.join(args.data_path, lang, "train"))
        elif args.data_type == "fineweb_removed":
            file_list = os.listdir(os.path.join(args.data_path, lang + "_removed", "train"))
        elif args.data_type == "mala":
            all_file_list = os.listdir(os.path.join(args.data_path, lang))
            file_list = [file_name for file_name in all_file_list if "arrow" in file_name]
        elif args.data_type == "nllb":
            file_list = os.listdir(os.path.join(args.data_path, lang))
        elif args.data_type == "oscar":
            file_list = os.listdir(os.path.join(args.data_path, "data", args.oscar_version, lang))
        else:
            print("data type currently not supported, please ask ....")
            break

        ## process type
        if args.process_type == "one":
            selected_file_list = random.sample(file_list, 1)
        elif args.process_type == "all":
            selected_file_list = sorted(file_list)
        elif args.process_type == "one_specific":
            selected_file_list = [args.one_specific_file]
        else:
            print("process type not supported, please ask")
            break
        
        ## 遍历该语言下的所有文件
        for file_name in selected_file_list:
            file_name_prefix = os.path.splitext(file_name)[0]
            ## 判断是否已经处理过，处理过了就直接跳过
            if args.data_type == "oscar":
                if os.path.exists(os.path.join(args.out_path, args.data_type, args.oscar_version, lang, f"{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_keep.jsonl")):
                    continue
            else:
                if os.path.exists(os.path.join(args.out_path, args.data_type, lang, f"{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_keep.jsonl")):
                    continue

            if args.data_type in ["culturax", "madlad"]:
                selected_file_path = os.path.join(args.data_path, GLOT_TO_ISO_1[lang], file_name)
            elif args.data_type == "fineweb":
                selected_file_path = os.path.join(args.data_path, lang, "train", file_name)
            elif args.data_type == "fineweb_removed":
                selected_file_path = os.path.join(args.data_path, lang + "_removed", "train", file_name)
            elif args.data_type in ["mala", "nllb"]:
                selected_file_path = os.path.join(args.data_path, lang, file_name)
            elif args.data_type == "oscar":
                selected_file_path = os.path.join(args.data_path, "data", args.oscar_version, lang, file_name)
            else:
                selected_file_path = os.path.join(args.data_path, lang, file_name)
            
            ## 加分布式锁
            from distributed_lock import lock_source_file, unlock_source_file
            if not lock_source_file(selected_file_path, time_out=2 * 3600):
                continue
        
            ## load dataset
            if "json" in file_name:
                dataset = load_dataset("json", data_files={"train": selected_file_path}, split="train", cache_dir=args.cache_dir)
            elif "parquet" in file_name:
                try:
                    dataset = load_dataset("parquet", data_files={"train": selected_file_path}, split="train", cache_dir=args.cache_dir)
                except:
                    print(f"File Broken: {selected_file_path}")
                    continue
            elif "arrow" in file_name:
                dataset = load_dataset("arrow", data_files={"train": selected_file_path}, split="train", cache_dir=args.cache_dir)
            else:
                print("load dataset should be updated, please ask.")


            ## step 1: modify the dataset (some special characters/ some normalization and others ... ...)
            modification_func = FunctionDatasetModifyingDocuments(lang)
            dataset = dataset.map(modification_func, num_proc=args.num_proc)
            print(f"Finished the Modification for {args.data_type}/{lang}/{file_name}")

            ## map the dataset --- generate the clean statistics
            dataset = dataset.map(statistic_mapping, num_proc=args.num_proc)

            ## statistic start time
            sta_start_time = time.time()
            ## write dict list
            write_dict_list = []

            ## 各项统计数据
            avg_dict = {}
            std_dict = {}
            median_dict = {}
            max_dict = {}
            min_dict = {}
            per_90_dict = {}
            per_75_dict = {}
            per_50_dict = {}
            per_25_dict = {}

            num_words_data = dataset["num_words"]
            character_repetition_ratio_data = dataset["character_repetition_ratio"]
            word_repetition_ratio_data = dataset["word_repetition_ratio"]
            special_characters_ratio_data = dataset["special_characters_ratio"]
            stopwords_ratio_data = dataset["stopwords_ratio"]
            flagged_words_ratio_data = dataset["flagged_words_ratio"]
            lang_id_score_data = dataset["lang_id_score"]
            perplexity_score_data = dataset["perplexity_score"]

            ## 均值
            avg_dict["num_words"] = float(np.mean(num_words_data))
            avg_dict["character_repetition_ratio"] = float(np.mean(character_repetition_ratio_data))
            avg_dict["word_repetition_ratio"] = float(np.mean(word_repetition_ratio_data))
            avg_dict["special_characters_ratio"] = float(np.mean(special_characters_ratio_data))
            avg_dict["stopwords_ratio"] = float(np.mean(stopwords_ratio_data))
            avg_dict["flagged_words_ratio"] = float(np.mean(flagged_words_ratio_data))
            avg_dict["lang_id_score"] = float(np.mean(lang_id_score_data))
            avg_dict["perplexity_score"] = float(np.mean(perplexity_score_data))
            
            ## 标准差
            std_dict["num_words"] = float(np.std(num_words_data))
            std_dict["character_repetition_ratio"] = float(np.std(character_repetition_ratio_data))
            std_dict["word_repetition_ratio"] = float(np.std(word_repetition_ratio_data))
            std_dict["special_characters_ratio"] = float(np.std(special_characters_ratio_data))
            std_dict["stopwords_ratio"] = float(np.std(stopwords_ratio_data))
            std_dict["flagged_words_ratio"] = float(np.std(flagged_words_ratio_data))
            std_dict["lang_id_score"] = float(np.std(lang_id_score_data))
            std_dict["perplexity_score"] = float(np.std(perplexity_score_data))

            ## 中位数
            median_dict["num_words"] = float(np.median(num_words_data))
            median_dict["character_repetition_ratio"] = float(np.median(character_repetition_ratio_data))
            median_dict["word_repetition_ratio"] = float(np.median(word_repetition_ratio_data))
            median_dict["special_characters_ratio"] = float(np.median(special_characters_ratio_data))
            median_dict["stopwords_ratio"] = float(np.median(stopwords_ratio_data))
            median_dict["flagged_words_ratio"] = float(np.median(flagged_words_ratio_data))
            median_dict["lang_id_score"] = float(np.median(lang_id_score_data))
            median_dict["perplexity_score"] = float(np.median(perplexity_score_data))

            ##最大值
            max_dict["num_words"] = float(np.max(num_words_data))
            max_dict["character_repetition_ratio"] = float(np.max(character_repetition_ratio_data))
            max_dict["word_repetition_ratio"] = float(np.max(word_repetition_ratio_data))
            max_dict["special_characters_ratio"] = float(np.max(special_characters_ratio_data))
            max_dict["stopwords_ratio"] = float(np.max(stopwords_ratio_data))
            max_dict["flagged_words_ratio"] = float(np.max(flagged_words_ratio_data))
            max_dict["lang_id_score"] = float(np.max(lang_id_score_data))
            max_dict["perplexity_score"] = float(np.max(perplexity_score_data))

            ## 最小值
            min_dict["num_words"] = float(np.min(num_words_data))
            min_dict["character_repetition_ratio"] = float(np.min(character_repetition_ratio_data))
            min_dict["word_repetition_ratio"] = float(np.min(word_repetition_ratio_data))
            min_dict["special_characters_ratio"] = float(np.min(special_characters_ratio_data))
            min_dict["stopwords_ratio"] = float(np.min(stopwords_ratio_data))
            min_dict["flagged_words_ratio"] = float(np.min(flagged_words_ratio_data))
            min_dict["lang_id_score"] = float(np.min(lang_id_score_data))
            min_dict["perplexity_score"] = float(np.min(perplexity_score_data))

            ## 分位数
            per_90_dict["num_words"] = float(np.percentile(num_words_data, 90))
            per_90_dict["character_repetition_ratio"] = float(np.percentile(character_repetition_ratio_data, 90))
            per_90_dict["word_repetition_ratio"] = float(np.percentile(word_repetition_ratio_data, 90))
            per_90_dict["special_characters_ratio"] = float(np.percentile(special_characters_ratio_data, 90))
            per_90_dict["stopwords_ratio"] = float(np.percentile(stopwords_ratio_data, 90))
            per_90_dict["flagged_words_ratio"] = float(np.percentile(flagged_words_ratio_data, 90))
            per_90_dict["lang_id_score"] = float(np.percentile(lang_id_score_data, 90))
            per_90_dict["perplexity_score"] = float(np.percentile(perplexity_score_data, 90))

            per_75_dict["num_words"] = float(np.percentile(num_words_data, 75))
            per_75_dict["character_repetition_ratio"] = float(np.percentile(character_repetition_ratio_data, 75))
            per_75_dict["word_repetition_ratio"] = float(np.percentile(word_repetition_ratio_data, 75))
            per_75_dict["special_characters_ratio"] = float(np.percentile(special_characters_ratio_data, 75))
            per_75_dict["stopwords_ratio"] = float(np.percentile(stopwords_ratio_data, 75))
            per_75_dict["flagged_words_ratio"] = float(np.percentile(flagged_words_ratio_data, 75))
            per_75_dict["lang_id_score"] = float(np.percentile(lang_id_score_data, 75))
            per_75_dict["perplexity_score"] = float(np.percentile(perplexity_score_data, 75))

            per_50_dict["num_words"] = float(np.percentile(num_words_data, 50))
            per_50_dict["character_repetition_ratio"] = float(np.percentile(character_repetition_ratio_data, 50))
            per_50_dict["word_repetition_ratio"] = float(np.percentile(word_repetition_ratio_data, 50))
            per_50_dict["special_characters_ratio"] = float(np.percentile(special_characters_ratio_data, 50))
            per_50_dict["stopwords_ratio"] = float(np.percentile(stopwords_ratio_data, 50))
            per_50_dict["flagged_words_ratio"] = float(np.percentile(flagged_words_ratio_data, 50))
            per_50_dict["lang_id_score"] = float(np.percentile(lang_id_score_data, 50))
            per_50_dict["perplexity_score"] = float(np.percentile(perplexity_score_data, 50))

            per_25_dict["num_words"] = float(np.percentile(num_words_data, 25))
            per_25_dict["character_repetition_ratio"] = float(np.percentile(character_repetition_ratio_data, 25))
            per_25_dict["word_repetition_ratio"] = float(np.percentile(word_repetition_ratio_data, 25))
            per_25_dict["special_characters_ratio"] = float(np.percentile(special_characters_ratio_data, 25))
            per_25_dict["stopwords_ratio"] = float(np.percentile(stopwords_ratio_data, 25))
            per_25_dict["flagged_words_ratio"] = float(np.percentile(flagged_words_ratio_data, 25))
            per_25_dict["lang_id_score"] = float(np.percentile(lang_id_score_data, 25))
            per_25_dict["perplexity_score"] = float(np.percentile(perplexity_score_data, 25))

            ## 记录统计值
            write_dict_list.append({"avg": avg_dict})
            write_dict_list.append({"std": std_dict})
            write_dict_list.append({"median": median_dict})
            write_dict_list.append({"max": max_dict})
            write_dict_list.append({"min": min_dict})
            write_dict_list.append({"90%": per_90_dict})
            write_dict_list.append({"75%": per_75_dict})
            write_dict_list.append({"50%": per_50_dict})
            write_dict_list.append({"25%": per_25_dict})

            ## statistic end time
            sta_end_time = time.time()
            sta_time = sta_end_time - sta_start_time
            print(f"统计用时：{sta_time}")

            stats_columns = [
                "num_words",
                "character_repetition_ratio",
                "word_repetition_ratio",
                "special_characters_ratio",
                "stopwords_ratio",
                "flagged_words_ratio",
                "lang_id_score",
                "perplexity_score"
            ]

            ## method time
            method_start_time = time.time()

            # Extract the statistics values
            stats_data = [[dataset[idx][item] for item in stats_columns] for idx in range(dataset.num_rows)]

            # Standardize the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(stats_data)

            if args.method_type == "kmeans":
                # Perform K-Means clustering
                kmeans = KMeans(n_clusters=2, random_state=42)  # 2 clusters: normal and anomalous
                clusters_detect = kmeans.fit_predict(scaled_data)
            elif args.method_type == "oc_svm":
                # Train one class svm
                one_svm = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
                one_svm.fit(scaled_data)
                clusters_detect = one_svm.predict(scaled_data)
            elif args.method_type == "iso_forest":
                iso_forest = IsolationForest(contamination="auto", random_state=42, n_jobs=args.num_proc)
                clusters_detect = iso_forest.fit_predict(scaled_data)
            elif args.method_type == "lof":
                lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1, n_jobs=args.num_proc)
                clusters_detect = lof.fit_predict(scaled_data)
            else:
                print("method type not support currently, please ask.")

            ## method ending time
            method_end_time = time.time()
            sta_method_time = method_end_time - method_start_time
            print(f"检测耗时: {sta_method_time}")

            ### draw the figure
            if args.draw_fig:
                draw_figure(stats_columns, stats_data, clusters_detect, args, lang, file_name)

            # Add cluster/detection labels to the dataset
            dataset = dataset.add_column("cluster_detection", clusters_detect)

            if args.method_type == "kmeans":
                keep_ds = dataset.filter(lambda x: x["cluster_detection"] == 0, num_proc=args.num_proc)  # Cluster 0 is assumed as "normal"
                remove_ds = dataset.filter(lambda x: x["cluster_detection"] == 1, num_proc=args.num_proc)  # Cluster 1 is assumed as "anomalous"
            elif args.method_type in ["oc_svm", "iso_forest", "lof"]:
                keep_ds = dataset.filter(lambda x: x["cluster_detection"] == 1, num_proc=args.num_proc)  # 1: normal
                remove_ds = dataset.filter(lambda x: x["cluster_detection"] == -1, num_proc=args.num_proc)  # -1: anomaly
            else:
                print("method type not supported, please ask.")
            
            write_dict_list.append({
                "total size": f"{dataset.num_rows}",
                "keep size": f"{keep_ds.num_rows}",
                "remove size": f"{remove_ds.num_rows}"
            })
            
            ## 写文件
            if args.data_type == "oscar":
                keep_ds.to_json(f"{args.out_path}/{args.data_type}/{args.oscar_version}/{lang}/{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_keep.jsonl", force_ascii=False)
                remove_ds.to_json(f"{args.out_path}/{args.data_type}/{args.oscar_version}/{lang}/{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_remove.jsonl", force_ascii=False)
                ## 统计文件
                with open(f"{args.out_path}/{args.data_type}/{args.oscar_version}/{lang}/{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_stas.jsonl", "w") as f:
                    json.dump(write_dict_list, f, indent=4)
                print(f"数据写入完毕")
            else:
                keep_ds.to_json(f"{args.out_path}/{args.data_type}/{lang}/{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_keep.jsonl", force_ascii=False)
                remove_ds.to_json(f"{args.out_path}/{args.data_type}/{lang}/{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_remove.jsonl", force_ascii=False)
                ## 统计文件
                with open(f"{args.out_path}/{args.data_type}/{lang}/{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_stas.jsonl", "w") as f:
                    json.dump(write_dict_list, f, indent=4)
                print(f"数据写入完毕")

            ## 释放分布式锁
            unlock_source_file(selected_file_path)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="select the filter parameters dynamiclly.")
    # parser.add_argument("--lang_list", type=str, nargs='+', help='language list to clean')
    # parser.add_argument("--character_repetition_length", type=int, default=10)
    # parser.add_argument("--draw_fig", action="store_true", help="draw the statistic figure?")
    # parser.add_argument("--num_proc", type=int, default=16)
    # parser.add_argument("--data_type", type=str, default="")
    # parser.add_argument("--process_type", type=str, default="")
    # parser.add_argument("--method_type", type=str, default="")
    # parser.add_argument("--cache_dir", type=str, default="")
    # parser.add_argument("--data_path", type=str, default="")
    # parser.add_argument("--out_path", type=str, default="")
    # parser.add_argument("--lid_path", type=str, default="")
    # parser.add_argument("--sp_path", type=str, default="")
    # parser.add_argument("--lm_path", type=str, default="")

    # args = parser.parse_args()
    main(args)