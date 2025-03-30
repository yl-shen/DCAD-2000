import argparse, os, random
import numpy as np
import json

from filtering import LoadParameters, ModifyingDocuments, Filtering

from datasets import load_dataset
from sklearn.cluster import KMeans

import fasttext
import sentencepiece
import kenlm

"""
python parameters_select.py \
--lang_list en zh de \
--num_proc 16 \
--data_type culturax \
--cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/code/dc_pipeline/.cache \
--data_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/huggingface/CulturaX \
--sp_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/sp_lm \
--out_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out/param_select

python parameters_select.py \
--lang_list cmn_Hani \
--num_proc 16 \
--data_type fineweb \
--cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/code/dc_pipeline/.cache \
--data_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/datasets/mono/fineweb-2/data \
--lm_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/sp_lm \
--sp_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/sp_lm \
--out_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out/param_select

#容器
python parameters_select.py \
--character_repetition_length 10 \
--num_proc 16 \
--lang_id en \
--data_path /data/huggingface/CulturaX/en \
--lid_path /data/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin \
--sp_path /data/shenyingli/hf_model/sp_lm/en.sp.model \
--lm_path /data/shenyingli/hf_model/sp_lm/en.arpa.bin

"""

current_file_path = os.path.abspath(__file__)
current_file_dir = os.path.dirname(current_file_path)
LID_PATH = os.path.join(current_file_dir, '..', '..', 'hf_model', 'fasttext-language-identification', 'glotlid', 'model_v3.bin')
## GLOT CONFIG PATH
GLOT_CONFIG_PATH = os.path.join(current_file_dir, "post_scripts", "lang_dict", "glot_mapping.csv")

GLOT_TO_ISO_1 = dict()
with open(GLOT_CONFIG_PATH, "r", encoding="utf-8") as f:
    for row in f:
        tmp_list = row.strip().split(",")
        GLOT_TO_ISO_1[tmp_list[0]] = tmp_list[2]

## load glot lid model
lid_model  = fasttext.load_model(LID_PATH)


def main(args):
    lang_list = args.lang_list
    for lang in lang_list:
        param = LoadParameters.load_parameters(lang)
        stopwords = LoadParameters.load_stopwords(lang)
        flagged_words = LoadParameters.load_flagged_words(lang)
        # lid_model = LoadParameters.load_model_lang_id(LID_PATH)
        if os.path.exists(f"{args.sp_path}/{GLOT_TO_ISO_1[lang]}.sp.model"):
            sp_model = LoadParameters.load_sentencepiece_model(f"{args.sp_path}/{GLOT_TO_ISO_1[lang]}.sp.model")
            lm_model = LoadParameters.load_kenlm_model(f"{args.lm_path}/{GLOT_TO_ISO_1[lang]}.arpa.bin")
        else:
            sp_model = None
            lm_model = None
        ## file_list
        if args.data_type in ["culturax", "madlad"]:
            file_list = os.listdir(os.path.join(args.data_path, GLOT_TO_ISO_1[lang]))
        elif args.data_type == "fineweb":
            file_list = os.listdir(os.path.join(args.data_path, lang, "train"))
        else:
            file_list = os.listdir(os.path.join(args.data_path, lang))
        if len(file_list) <= 2:
            selected_file_list = file_list
        else:
            selected_file_list = random.sample(file_list, 2)
        if args.data_type in ["culturax", "madlad"]:
            selected_file_path_list = [os.path.join(args.data_path, GLOT_TO_ISO_1[lang], file_name) for file_name in selected_file_list if 'parquet' in file_name or 'jsonl' in file_name]
        elif args.data_type == "fineweb":
            selected_file_path_list = [os.path.join(args.data_path, lang, "train", file_name) for file_name in selected_file_list if 'parquet' in file_name or 'jsonl' in file_name]
        else:
            selected_file_path_list = [os.path.join(args.data_path, lang, file_name) for file_name in selected_file_list if 'parquet' in file_name or 'jsonl' in file_name]
        if args.data_type == "madlad":
            dataset = load_dataset("json", data_files={"train": selected_file_path_list}, split="train", cache_dir=args.cache_dir)
        elif args.data_type in ["culturax", "glotcc", "fineweb"]:
            dataset = load_dataset("parquet", data_files={"train": selected_file_path_list}, split="train", cache_dir=args.cache_dir)
        
        
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

        ## map the dataset
        # dataset = dataset.map(lambda example: statistic_mapping(example, param, stopwords, flagged_words, sp_model, lm_model), num_proc=args.num_proc)
        dataset = dataset.map(statistic_mapping, num_proc=args.num_proc)
        # dataset = dataset.map(compute_lang_id_score, num_proc=args.num_proc)

        ## write dict list
        write_dict_list = []

        ## 各项统计数据
        avg_dict = {}
        avg_dict["num_words"] = sum(dataset["num_words"]) / dataset.num_rows
        avg_dict["character_repetition_ratio"] = sum(dataset["character_repetition_ratio"]) / dataset.num_rows
        avg_dict["word_repetition_ratio"] = sum(dataset["word_repetition_ratio"]) / dataset.num_rows
        avg_dict["special_characters_ratio"] = sum(dataset["special_characters_ratio"]) / dataset.num_rows
        avg_dict["stopwords_ratio"] = sum(dataset["stopwords_ratio"]) / dataset.num_rows
        avg_dict["flagged_words_ratio"] = sum(dataset["flagged_words_ratio"]) / dataset.num_rows
        avg_dict["lang_id_score"] = sum(dataset["lang_id_score"]) / dataset.num_rows
        avg_dict["perplexity_score"] = sum(dataset["perplexity_score"]) / dataset.num_rows
        
        print(avg_dict)
        write_dict_list.append(avg_dict)

        ## 无监督聚类
        num_words = dataset["num_words"]
        character_repetition_ratio = dataset["character_repetition_ratio"]
        word_repetition_ratio = dataset["word_repetition_ratio"]
        special_characters_ratio = dataset["special_characters_ratio"]
        stopwords_ratio = dataset["stopwords_ratio"]
        flagged_words_ratio = dataset["flagged_words_ratio"]
        lang_id_score = dataset["lang_id_score"]
        perplexity_score = dataset["perplexity_score"]

        features = np.stack([num_words, character_repetition_ratio, word_repetition_ratio, special_characters_ratio, stopwords_ratio, flagged_words_ratio, lang_id_score, perplexity_score], axis=1)
        # 1. 聚类 (K-Means)
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(features)

        # 2. 提取高质量簇的中心和范围
        high_quality_cluster = 0 if kmeans.cluster_centers_[0, 0] > kmeans.cluster_centers_[1, 0] else 1
        filtered_indices = np.where(labels == high_quality_cluster)[0]

        # 3. 提取阈值
        select_dict = {}

        filtered_features = features[filtered_indices]

        select_dict["num_words_threshold"] = filtered_features[:, 0].min()
        select_dict["character_repetition_ratio_threshold"] = filtered_features[:, 1].max()
        select_dict["word_repetition_ratio_threshold"] = filtered_features[:, 2].max()
        select_dict["special_characters_ratio_threshold"] = filtered_features[:, 3].max()
        select_dict["stopwords_ratio_threshold"] = filtered_features[:, 4].max()
        select_dict["flagged_words_ratio_threshold"] = filtered_features[:, 5].max()
        select_dict["lang_id_score_threshold"] = filtered_features[:, 6].max()
        select_dict["perplexity_score_threshold"] = filtered_features[:, 7].max()

        print(f'Selected thresholds - num_words: >= {select_dict["num_words_threshold"]}, character_repetition_ratio: >= {select_dict["character_repetition_ratio_threshold"]}, word_repetition_ratio: <= {select_dict["word_repetition_ratio_threshold"]}, special_characters_ratio: >= {select_dict["special_characters_ratio_threshold"]}, stopwords_ratio_threshold: >= {select_dict["stopwords_ratio_threshold"]}, flagged_words_ratio_threshold: >= {select_dict["flagged_words_ratio_threshold"]}, lang_id_score_threshold: >= {select_dict["lang_id_score_threshold"]}, perplexity_score_threshold: >= {select_dict["perplexity_score_threshold"]}')
        print(f"Filtered data size: {len(filtered_indices)} / {len(features)}")
        ## 
        write_dict_list.append(select_dict)
        write_dict_list.append({"filter_size": f"{len(filtered_indices)} / {len(features)}"})

        ## 确保目录存在
        os.makedirs(os.path.join(args.out_path, args.data_type), exist_ok=True)

        with open(f"{args.out_path}/{args.data_type}/{lang}_param.jsonl", 'w') as json_file:
            for json_dict in write_dict_list:
                json_file.write(json.dumps(json_dict) + '\n')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="select the filter parameters dynamiclly.")
    parser.add_argument("--lang_list", type=str, nargs='+', help='language list to clean')
    parser.add_argument("--character_repetition_length", type=int, default=10)
    parser.add_argument("--num_proc", type=int, default=16)
    parser.add_argument("--data_type", type=str, default="")
    parser.add_argument("--cache_dir", type=str, default="")
    parser.add_argument("--data_path", type=str, default="")
    parser.add_argument("--out_path", type=str, default="")
    parser.add_argument("--lid_path", type=str, default="")
    parser.add_argument("--sp_path", type=str, default="")
    parser.add_argument("--lm_path", type=str, default="")

    args = parser.parse_args()
    main(args)