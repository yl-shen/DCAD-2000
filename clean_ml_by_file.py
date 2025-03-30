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
--- data_type: fineweb/fineweb_removed/mala/nllb/oscar/      
--- method_type: kmeans / oc_svm / iso_forest /lof
--- lang_list: iso-639-3加上下划线加上语言script版本，多个语言的话，需要空格分开
--- draw_fig: 如果需要生成图片就加上 --draw_fig， 如果在集群里就不要加

传入文件名列表

python clean_ml_by_file.py \
--process_type all \
--lang_list eng_Latn \
--file_list *** \
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

### 下面是fineweb英文处理####

python clean_ml_by_file.py \
--process_type all \
--lang_list eng_Latn \
--file_list 000_00000.parquet 000_00001.parquet 000_00002.parquet 000_00003.parquet 000_00004.parquet 000_00005.parquet 000_00006.parquet 000_00007.parquet 000_00008.parquet 000_00009.parquet 000_00010.parquet 000_00011.parquet 000_00012.parquet 000_00013.parquet 000_00014.parquet 000_00015.parquet 000_00016.parquet 000_00017.parquet 000_00018.parquet 000_00019.parquet 000_00020.parquet 000_00021.parquet 000_00022.parquet 000_00023.parquet 000_00024.parquet 000_00025.parquet 000_00026.parquet 000_00027.parquet 000_00028.parquet 000_00029.parquet 000_00030.parquet 000_00031.parquet 000_00032.parquet 000_00033.parquet 000_00034.parquet 000_00035.parquet 000_00036.parquet 000_00037.parquet 000_00038.parquet 000_00039.parquet 000_00040.parquet 000_00041.parquet 000_00042.parquet 000_00043.parquet 000_00044.parquet 000_00045.parquet 000_00046.parquet 000_00047.parquet 000_00048.parquet 000_00049.parquet 001_00000.parquet 001_00001.parquet 001_00002.parquet 001_00003.parquet 001_00004.parquet 001_00005.parquet 001_00006.parquet 001_00007.parquet 001_00008.parquet 001_00009.parquet 001_00010.parquet 001_00011.parquet 001_00012.parquet 001_00013.parquet 001_00014.parquet 001_00015.parquet 001_00016.parquet 001_00017.parquet 001_00018.parquet 001_00019.parquet 001_00020.parquet 001_00021.parquet 001_00022.parquet 001_00023.parquet 001_00024.parquet 001_00025.parquet 001_00026.parquet 001_00027.parquet 001_00028.parquet 001_00029.parquet 001_00030.parquet 001_00031.parquet 001_00032.parquet 001_00033.parquet 001_00034.parquet 001_00035.parquet 001_00036.parquet 001_00037.parquet 001_00038.parquet 001_00039.parquet 001_00040.parquet 001_00041.parquet 001_00042.parquet 001_00043.parquet 001_00044.parquet 001_00045.parquet 001_00046.parquet 001_00047.parquet 001_00048.parquet 001_00049.parquet 002_00000.parquet 002_00001.parquet 002_00002.parquet 002_00003.parquet 002_00004.parquet 002_00005.parquet 002_00006.parquet 002_00007.parquet 002_00008.parquet 002_00009.parquet 002_00010.parquet 002_00011.parquet 002_00012.parquet 002_00013.parquet 002_00014.parquet 002_00015.parquet 002_00016.parquet 002_00017.parquet 002_00018.parquet 002_00019.parquet 002_00020.parquet 002_00021.parquet 002_00022.parquet 002_00023.parquet 002_00024.parquet 002_00025.parquet 002_00026.parquet 002_00027.parquet 002_00028.parquet 002_00029.parquet 002_00030.parquet 002_00031.parquet 002_00032.parquet 002_00033.parquet 002_00034.parquet 002_00035.parquet 002_00036.parquet 002_00037.parquet 002_00038.parquet 002_00039.parquet 002_00040.parquet 002_00041.parquet 002_00042.parquet 002_00043.parquet 002_00044.parquet 002_00045.parquet 002_00046.parquet 002_00047.parquet 002_00048.parquet 002_00049.parquet 003_00000.parquet 003_00001.parquet 003_00002.parquet 003_00003.parquet 003_00004.parquet 003_00005.parquet 003_00006.parquet 003_00007.parquet 003_00008.parquet 003_00009.parquet 003_00010.parquet 003_00011.parquet 003_00012.parquet 003_00013.parquet 003_00014.parquet 003_00015.parquet 003_00016.parquet 003_00017.parquet 003_00018.parquet 003_00019.parquet 003_00020.parquet 003_00021.parquet 003_00022.parquet 003_00023.parquet 003_00024.parquet 003_00025.parquet 003_00026.parquet 003_00027.parquet 003_00028.parquet 003_00029.parquet 003_00030.parquet 003_00031.parquet 003_00032.parquet 003_00033.parquet 003_00034.parquet 003_00035.parquet 003_00036.parquet 003_00037.parquet 003_00038.parquet 003_00039.parquet 003_00040.parquet 003_00041.parquet 003_00042.parquet 003_00043.parquet 003_00044.parquet 003_00045.parquet 003_00046.parquet 003_00047.parquet 003_00048.parquet 003_00049.parquet 004_00000.parquet 004_00001.parquet 004_00002.parquet 004_00003.parquet 004_00004.parquet 004_00005.parquet 004_00006.parquet 004_00007.parquet 004_00008.parquet 004_00009.parquet 004_00010.parquet 004_00011.parquet 004_00012.parquet 004_00013.parquet 004_00014.parquet 004_00015.parquet 004_00016.parquet 004_00017.parquet 004_00018.parquet 004_00019.parquet 004_00020.parquet 004_00021.parquet 004_00022.parquet 004_00023.parquet 004_00024.parquet 004_00025.parquet 004_00026.parquet 004_00027.parquet 004_00028.parquet 004_00029.parquet 004_00030.parquet 004_00031.parquet 004_00032.parquet 004_00033.parquet 004_00034.parquet 004_00035.parquet 004_00036.parquet 004_00037.parquet 004_00038.parquet 004_00039.parquet 004_00040.parquet 004_00041.parquet 004_00042.parquet 004_00043.parquet 004_00044.parquet 004_00045.parquet 004_00046.parquet 004_00047.parquet 004_00048.parquet 004_00049.parquet \
--num_proc 28 \
--data_type fineweb_en \
--method_type iso_forest \
--cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/.cache/fineweb-en \
--data_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/datasets/mono/CC-MAIN-2024-46-fineweb_en \
--lid_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin \
--lm_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_lm \
--sp_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_sp \
--out_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out_colony



------------paper-------
python clean_ml_by_file.py \
--draw_fig \
--process_type all \
--lang_list cmn_Hani \
--file_list 000_00011.parquet \
--num_proc 28 \
--data_type fineweb \
--method_type iso_forest \
--cache_dir /mb-datacenter/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/.cache \
--data_path /mb-datacenter/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/datasets/mono/fineweb-2/data \
--lid_path /mb-datacenter/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin \
--lm_path /mb-datacenter/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_lm \
--sp_path /mb-datacenter/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_sp \
--out_path /mb-datacenter/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/acl_paper


"""

current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)

## 设置参数
parser = argparse.ArgumentParser(description="select the filter parameters dynamiclly.")
parser.add_argument("--lang_list", type=str, nargs='+', help='language list to clean')
parser.add_argument("--file_list", type=str, nargs='+', help='language list to clean')
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
    plt.suptitle("Anomaly Detecting Results: High vs Low Quality Data", fontsize=16)
    if args.data_type == "oscar":
        plt.savefig(f"{args.out_path}/{args.data_type}/{args.oscar_version}/{lang}/{args.data_type}_{args.method_type}_{lang}_{os.path.splitext(file_name)[0]}.png")
    else:
        plt.savefig(f"{args.out_path}/{args.data_type}/{lang}/{args.data_type}_{args.method_type}_{lang}_{os.path.splitext(file_name)[0]}.png")
'''
    if args.data_type == "oscar":
        plt.savefig(f"{args.out_path}/{args.data_type}/{args.oscar_version}/{lang}/{args.data_type}_{args.method_type}_{lang}_{os.path.splitext(file_name)[0]}.pdf")
    else:
        plt.savefig(f"{args.out_path}/{args.data_type}/{lang}/{args.data_type}_{args.method_type}_{lang}_{os.path.splitext(file_name)[0]}.pdf")
'''

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
        # if args.data_type in ["culturax", "madlad"]:
        #     file_list = os.listdir(os.path.join(args.data_path, GLOT_TO_ISO_1[lang]))
        # elif args.data_type == "fineweb":
        #     file_list = os.listdir(os.path.join(args.data_path, lang, "train"))
        # elif args.data_type == "fineweb_removed":
        #     file_list = os.listdir(os.path.join(args.data_path, lang + "_removed", "train"))
        # elif args.data_type == "mala":
        #     all_file_list = os.listdir(os.path.join(args.data_path, lang))
        #     file_list = [file_name for file_name in all_file_list if "arrow" in file_name]
        # elif args.data_type == "nllb":
        #     file_list = os.listdir(os.path.join(args.data_path, lang))
        # elif args.data_type == "oscar":
        #     file_list = os.listdir(os.path.join(args.data_path, "data", args.oscar_version, lang))
        # else:
        #     print("data type currently not supported, please ask ....")
        #     break
        file_list = args.file_list

        ## process type
        if args.process_type == "one":
            selected_file_list = random.sample(file_list, 1)
        elif args.process_type == "all":
            selected_file_list = sorted(file_list)
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
            elif args.data_type == "fineweb_en":
                selected_file_path = os.path.join(args.data_path, lang, file_name)
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