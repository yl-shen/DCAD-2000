import argparse, os
import matplotlib.pyplot as plt
import numpy as np
import random
import json
import time

from filtering import LoadParameters, ModifyingDocuments, Filtering, FunctionDatasetModifyingDocuments

from datasets import load_dataset
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor

import fasttext

"""

python dcad_by_file.py \
--process_type all \
--lang_list eng_Latn \
--file_list *** \
--num_proc 28 \
--method_type iso_forest \
--cache_dir ** \
--data_path ** \
--lid_path **/glotlid/model_v3.bin \
--lm_path ** \
--sp_path ** \
--out_path **

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
    plt.savefig(f"{args.out_path}/{args.data_type}/{lang}/{args.data_type}_{args.method_type}_{lang}_{os.path.splitext(file_name)[0]}.png")

VALID_SUFFIX = ["json", "parquet", "arrow", "txt"]

def main(args):
    lang_list = args.lang_list
    for lang in lang_list:
        os.makedirs(os.path.join(args.out_path, args.data_type, lang), exist_ok=True)

        param = LoadParameters.load_parameters(lang)
        stopwords = LoadParameters.load_stopwords(lang)
        flagged_words = LoadParameters.load_flagged_words(lang)

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

        
        file_list = args.file_list

        ## process type
        if args.process_type == "one":
            selected_file_list = random.sample(file_list, 1)
        elif args.process_type == "all":
            selected_file_list = sorted(file_list)
        else:
            print("process type not supported, please ask")
            break
        
        for file_name in selected_file_list:
            file_name_prefix = os.path.splitext(file_name)[0]
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
            elif args.data_type == "fineweb_en":
                selected_file_path = os.path.join(args.data_path, lang, file_name)
            else:
                selected_file_path = os.path.join(args.data_path, lang, file_name)
            
            ## distribution lock
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

            ## all sta dict
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

            ## mean
            avg_dict["num_words"] = float(np.mean(num_words_data))
            avg_dict["character_repetition_ratio"] = float(np.mean(character_repetition_ratio_data))
            avg_dict["word_repetition_ratio"] = float(np.mean(word_repetition_ratio_data))
            avg_dict["special_characters_ratio"] = float(np.mean(special_characters_ratio_data))
            avg_dict["stopwords_ratio"] = float(np.mean(stopwords_ratio_data))
            avg_dict["flagged_words_ratio"] = float(np.mean(flagged_words_ratio_data))
            avg_dict["lang_id_score"] = float(np.mean(lang_id_score_data))
            avg_dict["perplexity_score"] = float(np.mean(perplexity_score_data))
            
            ## std
            std_dict["num_words"] = float(np.std(num_words_data))
            std_dict["character_repetition_ratio"] = float(np.std(character_repetition_ratio_data))
            std_dict["word_repetition_ratio"] = float(np.std(word_repetition_ratio_data))
            std_dict["special_characters_ratio"] = float(np.std(special_characters_ratio_data))
            std_dict["stopwords_ratio"] = float(np.std(stopwords_ratio_data))
            std_dict["flagged_words_ratio"] = float(np.std(flagged_words_ratio_data))
            std_dict["lang_id_score"] = float(np.std(lang_id_score_data))
            std_dict["perplexity_score"] = float(np.std(perplexity_score_data))

            ## median
            median_dict["num_words"] = float(np.median(num_words_data))
            median_dict["character_repetition_ratio"] = float(np.median(character_repetition_ratio_data))
            median_dict["word_repetition_ratio"] = float(np.median(word_repetition_ratio_data))
            median_dict["special_characters_ratio"] = float(np.median(special_characters_ratio_data))
            median_dict["stopwords_ratio"] = float(np.median(stopwords_ratio_data))
            median_dict["flagged_words_ratio"] = float(np.median(flagged_words_ratio_data))
            median_dict["lang_id_score"] = float(np.median(lang_id_score_data))
            median_dict["perplexity_score"] = float(np.median(perplexity_score_data))

            ##maximum
            max_dict["num_words"] = float(np.max(num_words_data))
            max_dict["character_repetition_ratio"] = float(np.max(character_repetition_ratio_data))
            max_dict["word_repetition_ratio"] = float(np.max(word_repetition_ratio_data))
            max_dict["special_characters_ratio"] = float(np.max(special_characters_ratio_data))
            max_dict["stopwords_ratio"] = float(np.max(stopwords_ratio_data))
            max_dict["flagged_words_ratio"] = float(np.max(flagged_words_ratio_data))
            max_dict["lang_id_score"] = float(np.max(lang_id_score_data))
            max_dict["perplexity_score"] = float(np.max(perplexity_score_data))

            ## minimum
            min_dict["num_words"] = float(np.min(num_words_data))
            min_dict["character_repetition_ratio"] = float(np.min(character_repetition_ratio_data))
            min_dict["word_repetition_ratio"] = float(np.min(word_repetition_ratio_data))
            min_dict["special_characters_ratio"] = float(np.min(special_characters_ratio_data))
            min_dict["stopwords_ratio"] = float(np.min(stopwords_ratio_data))
            min_dict["flagged_words_ratio"] = float(np.min(flagged_words_ratio_data))
            min_dict["lang_id_score"] = float(np.min(lang_id_score_data))
            min_dict["perplexity_score"] = float(np.min(perplexity_score_data))

            ## per
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

            ## statistic record
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
            print(f"Time Cost: {sta_time}")

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
            print(f"Time Cost: {sta_method_time}")

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
            
            keep_ds.to_json(f"{args.out_path}/{args.data_type}/{lang}/{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_keep.jsonl", force_ascii=False)
            remove_ds.to_json(f"{args.out_path}/{args.data_type}/{lang}/{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_remove.jsonl", force_ascii=False)
            ## 统计文件
            with open(f"{args.out_path}/{args.data_type}/{lang}/{args.data_type}_{args.method_type}_{lang}_{file_name_prefix}_stas.jsonl", "w") as f:
                json.dump(write_dict_list, f, indent=4)
            print(f"数据写入完毕")

            ## 释放分布式锁
            unlock_source_file(selected_file_path)


if __name__ == "__main__":
    # args = parser.parse_args()
    main(args)