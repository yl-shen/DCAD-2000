method_type=("kmeans" "oc_svm" "iso_forest" "lof")

for method in "${method_type[@]}"
do
    echo "method type: $method"
    python clean_ml.py \
    --draw_fig \
    --process_type one \
    --lang_list cmn_Hani \
    --num_proc 28 \
    --data_type fineweb \
    --method_type $method \
    --cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/code/dc_pipeline/.cache \
    --data_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/datasets/mono/fineweb-2/data \
    --lm_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_lm \
    --sp_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_sp \
    --out_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out/data_clean_ml
done


#单独在llm环境下测试4种方法
python clean_ml.py \
--draw_fig \
--process_type one \
--lang_list cmn_Hani \
--num_proc 28 \
--data_type fineweb \
--method_type oc_svm \
--cache_dir /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/.cache \
--data_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/datasets/mono/fineweb-2/data \
--lid_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/fasttext-language-identification/glotlid/model_v3.bin \
--lm_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_lm \
--sp_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/hf_model/trained_sp \
--out_path /kfs-crawl/kfs-0437ff36-c86e-4fa8-949e-f64d1ca50d3e/shenyingli/data_clean_out/test_clean_ml

