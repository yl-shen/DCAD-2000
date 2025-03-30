## Dataset Summary
DCAD-2000 is a large-scale multilingual corpus built using
newly extracted Common Crawl data (CC-MAIN-2024-46) and existing multilingual datasets. It includes over 2,282 languages, 46.72TB of data, and 8.63 billion documents, spanning 155 highand medium-resource languages and 159 writing scripts. We propose reframing data cleaning as an anomaly detection task. This dynamic filtering approach significantly enhances data quality by identifying and removing noisy or anomalous content.

+ Paper: [A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection](https://www.arxiv.org/abs/2502.11546)
+ Github: [https://github.com/yl-shen/DCAD-2000](https://github.com/yl-shen/DCAD-2000)
+ Dataset (HuggingFace): [openbmb/DCAD-2000](https://huggingface.co/datasets/openbmb/DCAD-2000)

## Dataset Overview
Comparison of multilingual datasets constructed from Common Crawl (CC) and our constructed DCAD-2000, focusing on the latest CC version used, the total number of languages supported, distribution across resource categories (high, medium, low, very low), and training readiness. The CC version marked with **bold** indicates an inferred version due to the lack of explicit specification in the original paper. The ``Training-Ready'' column indicates whether the dataset is ready for training LLMs without requiring further data cleaning.


| **Dataset**               | **CC Version**     | **#Langs (total)** | **#Langs (high)** | **#Langs (medium)** | **#Langs (low)** | **#Langs (very low)** | **Training-Ready** |
|---------------------------|--------------------|--------------------|-------------------|---------------------|------------------|-----------------------|--------------------|
| mC4 (Raffel et al., 2020)  | CC-MAIN-2020-34    | 101                | 0                 | 43                  | 52               | 6                     | ✘                  |
| OSCAR 23.01 (Abadji et al., 2022) | CC-MAIN-2022-49    | 153                | 6                 | 42                  | 25               | 80                    | ✘                  |
| Glot500 (Imani et al., 2023) | **CC-MAIN-2020-34** | 511                | 0                 | 108                 | 79               | 324                   | ✘                  |
| CulturaX (Nguyen et al., 2024) | **CC-MAIN-2022-49** | 167                | 11                | 47                  | 27               | 82                    | ✘                  |
| Madlad-400 (Kudugunta et al., 2024) | CC-MAIN-2022-33    | 419                | 7                 | 46                  | 39               | 327                   | ✘                  |
| MaLA (Ji et al., 2024)         | **CC-MAIN-2022-49** | 939                | 1                 | 125                 | 78               | 735                   | ✘                  |
| Glotcc (Kargaran et al., 2024) | CC-MAIN-2023-50    | 1331               | 0                 | 10                  | 52               | 1269                  | ✘                  |
| HPLT-v1.2 (de Gilbert et al., 2024) | **CC-MAIN-2022-40** | 191                | 12                | 53                  | 38               | 88                    | ✘                  |
| Fineweb-2 (Penedo et al., 2024) | CC-MAIN-2024-18    | 1915               | 10                | 62                  | 49               | 1794                  | ✘                  |
| **DCAD-2000**             | CC-MAIN-2024-46    | 2282               | 13                | 142                 | 124              | 2003                  | ✓                  |

## Dataset Creation
+ **Data Collection:** DCAD-2000 integrates data from four main sources: MaLA, Fineweb, Fineweb-2, and newly extracted Common Crawl data.
+ **Data Cleaning as Anomaly Detection:** Traditional data cleaning methods rely on fixed thresholds for document-level features, making them less adaptable to the diversity of multilingual data. To address this, we propose a novel framework that formulates data cleaning as an anomaly detection task, which involves the feature extraction and anomaly detection.
    - **Feature Extraction:** For each document, we consider the following eight features: (1) Number of Words; (2) Character Repetition Ratio; (3) Word Repetition Ratio; (4) Special Characters Ratio; (5) Stop- words Ratio; (6) Flagged Words Ratio; (7) Language Identification (LID) Score; (8) Perplexity Score.
    - **Anomaly Detection:** We evaluate several classical anomaly detection algorithms including (1) Isolation Forest; (2) One Class SVM; (3) Local Outlier Factor and (4) K-Means.
    - Visualization<br>
    ![ad_overview](https://raw.githubusercontent.com/yl-shen/DCAD-2000/main/images/logo.png)

## Data Statistics
**传完github**再总结

## Usage
```
from datasets import load_dataset
data = load_dataset("openbmb/DCAD-2000")
```
You can also specifiy the language you wanted
```
from datasets import load_dataset
data = load_dataset("openbmb/DCAD-2000", lang="eng_Latn")
```


## Citation Information
```
@article{shen2025dcad,
  title={DCAD-2000: A Multilingual Dataset across 2000+ Languages with Data Cleaning as Anomaly Detection},
  author={Shen, Yingli and Lai, Wen and Wang, Shuo and Zhang, Xueren and Luo, Kangyang and Fraser, Alexander and Sun, Maosong},
  journal={arXiv preprint arXiv:2502.11546},
  year={2025}
}
```

## Acknowledgements
We introduce DCAD-2000, a large- scale multilingual dataset designed to address the increasing demand for high-quality and diverse training data for multilingual LLMs.
This work is done by researchers at [Tsinghua NLP group](https://huggingface.co/thunlp) in collaboration with partners from [TUM](https://wenlai-lavine.github.io/) and [Modelbest Inc.](https://huggingface.co/openbmb) 

## Contact Information
Yingli Shen (syl@mail.tsinghua.edu.cn)
Wen Lai (wen.lai@tum.de)

**加上logo**