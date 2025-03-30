### 以下是指定一个文件名取获取pdf的例子，图1
nohup python clean_ml_by_file.py \
--draw_fig \
--process_type all \
--lang_list cmn_Hani \
--file_list 000_00009.parquet \
--num_proc 32 \
--data_type fineweb \
--method_type iso_forest \
--cache_dir /data/shenyingli/.cache \
--data_path /data/shenyingli/datasets/mono/fineweb-2/data \
--lid_path /data/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin \
--lm_path /data/shenyingli/hf_model/trained_lm \
--sp_path /data/shenyingli/hf_model/trained_sp \
--out_path /data/shenyingli/acl_paper &

