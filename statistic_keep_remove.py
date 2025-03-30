"""
统计删除和保留的文本数量token: 文件统计/语言统计


python statistic_keep_remove.py \
--num_proc 16 \
--data_type culturax \
--lang_list ar de es fr it ja ko ru th zh \
--cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/.cache \
--orig_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/huggingface/CulturaX \
--filter_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out/culturax \
--statistic_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/statistics/test



"""

import argparse, os, csv, sys
import hashlib
from tqdm import tqdm
from datasets import load_dataset
# 计算 token 数量
from sacremoses import MosesTokenizer
# 禁用缓存
from datasets import disable_caching
disable_caching()


## language_ids path
file_dir = os.path.dirname(__file__)
sys.path.append(os.path.abspath(os.path.join(file_dir, '..', '..')))
# sys.path.append("/kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/code/dc_pipeline")
from languages_id import CULTURAX_LANG_LIST, MADLAD_LANG_LIST


def has_header(csv_file_path):
    if os.path.isfile(csv_file_path):
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            # 创建 CSV 阅读器
            reader = csv.reader(csvfile)
            # 读取第一行
            for row in reader:
                # 判断第一行是否包含表头 # 这里假设表头为非数字的字段名称，可以根据具体情况调整条件
                if all(isinstance(item, str) for item in row):
                    return True
                else:
                    return False
    else:
        return False

def get_already_statistic_file(csv_file_path, column_name):
    if os.path.isfile(csv_file_path):
        with open(csv_file_path, newline='', encoding='utf-8') as csvfile:
            # 创建 CSV 阅读器
            reader = csv.DictReader(csvfile)
            ## 获取指定列的值
            column_values = [row[column_name] for row in reader]
            return column_values
    else:
        return []


def main(args):
    all_file_header = ['File_Name', 'Total(sents)', 'Total(tokens)', 'Total(avg_tokens)', 'Total(size)', 'Keep(sents)', 'Keep(tokens)', 'Keep(avg_tokens)', 'Keep(size)', 'Remove(sents)', 'Remove(tokens)', 'Remove(avg_tokens)', 'Remove(size)']
    all_lang_header = ['Language', 'Total(sents)', 'Total(tokens)', 'Total(avg_tokens)', 'Total(size)', 'Keep(sents)', 'Keep(tokens)', 'Keep(avg_tokens)', 'Keep(size)', 'Remove(sents)', 'Remove(tokens)', 'remove(avg_tokens)' 'Remove(size)']
    
    all_lang_has_header = has_header(os.path.join(args.statistic_path, f"{args.data_type}_all_lang_statistic.csv"))

    all_lang_statistic_csv = open(os.path.join(args.statistic_path, f"{args.data_type}_all_lang_statistic.csv"), mode='a+', newline='')
    all_lang_writer = csv.writer(all_lang_statistic_csv, delimiter=",")
    
    ## 没有表头
    if not all_lang_has_header:
        all_lang_writer.writerow(all_lang_header)    

    ALL_LANG_LIST = args.lang_list

    for lang in tqdm(ALL_LANG_LIST, desc="Process Langguage: "):
        ## 获取已经统计的文件名
        finished_statistic_file_list = get_already_statistic_file(os.path.join(args.statistic_path, f"{args.data_type}_{lang}_all_file_statistic.csv"), "File_Name")
        lang_has_header = has_header(os.path.join(args.statistic_path, f"{args.data_type}_{lang}_all_file_statistic.csv"))

        all_file_statistic_csv = open(os.path.join(args.statistic_path, f"{args.data_type}_{lang}_all_file_statistic.csv"), mode='a+', newline='')
        all_file_writer = csv.writer(all_file_statistic_csv, delimiter=",")
        ## 判断是否有表头
        if not lang_has_header:
            all_file_writer.writerow(all_file_header)

        ## 加载分词器
        moses_tokenizer = MosesTokenizer(lang=lang)

        def hash_text(example):
            text = example["text"]
            if len(text) >= 100:
                hash_text = hashlib.sha256(text[:100].encode('utf-8')).hexdigest()
            else:
                hash_text = hashlib.sha256(text.encode('utf-8')).hexdigest()
            example["hash_text"] = hash_text
            return example

        def original_token_keep_remove(example, hash_set, moses_tokenizer):
            text = example["text"]
            token_list = moses_tokenizer.tokenize(text)
            example["token_count"] = len(token_list)
            # hash text
            if len(text) >= 100:
                hash_text = hashlib.sha256(text[:100].encode('utf-8')).hexdigest()
            else:
                hash_text = hashlib.sha256(text.encode('utf-8')).hexdigest()
            example["hash_text"] = hash_text
            if hash_text in hash_set:
                example["keep"] = "True"
            else:
                example["keep"] = "False"
            return example

        # all_file_list = sorted(os.listdir(os.path.join(args.orig_path, lang)))
        all_file_list = []
        for afl in os.listdir(os.path.join(args.orig_path, lang)):
            if afl.endswith("parquet") or afl.endswith("jsonl"):
                all_file_list.append(afl)
            else:
                continue
        all_file_list = sorted(all_file_list)

        if finished_statistic_file_list:
            unfinished_file_list = sorted(list(set(all_file_list) - set(finished_statistic_file_list)))
        else:
            unfinished_file_list = all_file_list
        
        ## statistic list (all file list)
        total_sents_list = []
        total_tokens_list = []
        total_avg_tokens_list = []
        total_size_list = []
        keep_sents_list = []
        keep_tokens_list = []
        keep_avg_tokens_list = []
        keep_size_list = []
        remove_sents_list = []
        remove_tokens_list = []
        remove_avg_tokens_list = []
        remove_size_list = []

        for file_name in unfinished_file_list:
            from distributed_lock import lock_source_file, unlock_source_file
            if not lock_source_file(file_name, time_out=1 * 3600):
                continue
            # 读取文件
            if args.data_type == "culturax":
                original_dataset = load_dataset('parquet', data_files = {'train': os.path.join(args.orig_path, f'{lang}', file_name)}, split='train', cache_dir=args.cache_dir)
                keep_dataset = load_dataset('json', data_files = {'train': os.path.join(args.filter_path, f'{lang}', f'{file_name}.keep.jsonl')}, split='train', cache_dir=args.cache_dir)
            elif args.data_type == "madlad":
                original_dataset = load_dataset('json', data_files = {'train': os.path.join(args.orig_path, f'{lang}', file_name)}, split='train', cache_dir=args.cache_dir)
                keep_dataset = load_dataset('json', data_files = {'train': os.path.join(args.filter_path, lang, file_name.replace('.jsonl', '') + '.keep.jsonl')}, split='train', cache_dir=args.cache_dir)
            # 统计
            total_sents = original_dataset.num_rows
            keep_sents = keep_dataset.num_rows
            remove_sents = total_sents - keep_sents

            total_size = original_dataset.data.nbytes
            keep_size = keep_dataset.data.nbytes
            remove_size = total_size - keep_size
            ## 对keep的text进行hash
            keep_hash_dataset = keep_dataset.map(hash_text, num_proc=args.num_proc)
            keep_hash_set = set(keep_hash_dataset["hash_text"])
            ## token，并标记keep
            original_token_dataset = original_dataset.map(lambda example: original_token_keep_remove(example, keep_hash_set, moses_tokenizer), num_proc=args.num_proc)

            ## 分词统计 (平均数和总数)
            total_token = sum(original_token_dataset["token_count"])
            total_avg_token = int(total_token / original_token_dataset.num_rows)

            keep_token_dataset = original_token_dataset.filter(lambda example: example["keep"] == "True")
            keep_token = sum(keep_token_dataset["token_count"])
            keep_avg_token = int(keep_token / keep_token_dataset.num_rows)


            remove_token = total_token - keep_token
            remove_avg_token = int(remove_token / remove_sents)

            ## 写入一个文件
            all_file_writer.writerow([file_name, total_sents, total_token, total_avg_token, total_size, keep_sents, keep_token, keep_avg_token, keep_size, remove_sents, remove_token, remove_avg_token, remove_size])
            all_file_statistic_csv.flush()

            ## 加入同一个语言的统计
            total_sents_list.append(total_sents)
            total_tokens_list.append(total_token)
            total_avg_tokens_list.append(total_avg_token)
            total_size_list.append(total_size)

            keep_sents_list.append(keep_sents)
            keep_tokens_list.append(keep_token)
            keep_avg_tokens_list.append(keep_avg_token)
            keep_size_list.append(keep_size)

            remove_sents_list.append(remove_sents)
            remove_tokens_list.append(remove_token)
            remove_avg_tokens_list.append(remove_avg_token)
            remove_size_list.append(remove_size)

            ## 清除缓存
            original_dataset.cleanup_cache_files()
            keep_dataset.cleanup_cache_files()
            keep_hash_dataset.cleanup_cache_files()
            original_token_dataset.cleanup_cache_files()
            keep_token_dataset.cleanup_cache_files()
            
            ## 解开分布式锁
            unlock_source_file(file_name)
        
        ## 一种语言遍历完毕，统计该语言的数据并写入文件
        lang_total_sents = sum(total_sents_list)
        lang_total_tokens = sum(total_tokens_list)
        lang_total_avg_tokens = sum(total_avg_tokens_list) / len(total_avg_tokens_list)
        lang_total_size = sum(total_size_list)

        lang_keep_sents = sum(keep_sents_list)
        lang_keep_tokens = sum(keep_tokens_list)
        lang_keep_avg_tokens = sum(keep_avg_tokens_list) / len (keep_avg_tokens_list)
        lang_keep_size = sum(keep_size_list)

        lang_remove_sents = sum(remove_sents_list)
        lang_remove_tokens = sum(remove_tokens_list)
        lang_remove_avg_tokens = sum(remove_avg_tokens_list) / len (remove_avg_tokens_list)
        lang_remove_size = sum(remove_size_list)
        ## 写入同一种语言的文件
        all_lang_writer.writerow([lang, lang_total_sents, lang_total_tokens, lang_total_avg_tokens, lang_total_size, lang_keep_sents, lang_keep_tokens, lang_keep_avg_tokens, lang_keep_size, lang_remove_sents, lang_remove_tokens, lang_remove_avg_tokens, lang_remove_size])
        all_lang_statistic_csv.flush()
        ## 所有文件可以追加一个总和
        all_file_writer.writerow([lang, lang_total_sents, lang_total_tokens, lang_total_avg_tokens, lang_total_size, lang_keep_sents, lang_keep_tokens, lang_keep_avg_tokens, lang_keep_size, lang_remove_sents, lang_remove_tokens, lang_remove_avg_tokens, lang_remove_size])
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filtering.")
    parser.add_argument(
        "--num_proc",
        type=int,
        default=16,
        help="number of process",
    )
    parser.add_argument(
        "--data_type",
        type=str,
        default="culturax",
        help="madlad or culturax",
    )
    parser.add_argument(
        "--orig_path",
        type=str,
        default="/kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/huggingface/CulturaX",
        help="original dataset path",
    )
    parser.add_argument(
        "--filter_path",
        type=str,
        default="/kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out/culturax",
        help="filter dataset path",
    )
    parser.add_argument(
        "--statistic_path",
        type=str,
        default="/kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/code/dc_pipeline/post_scripts/statistics/statis_res",
        help="statistic path",
    )
    parser.add_argument("--lang_list", type=str, nargs='+', help='language list to clean')
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="",
        help="cache path",
    )
    
    args = parser.parse_args()
    main(args)
