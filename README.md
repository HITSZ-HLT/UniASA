# UniASA: A Unified Generative Framework for Argument Structure Analysis

Jianzhu Bao, Mohan Jing, Kuicai Dong, Aixin Sun, Yang Sun, Ruifeng Xu.

*Computational Linguistics journal (2025)*

## Abstract

Argumentation is a fundamental human activity that involves reasoning and persuasion, which also serves as the basis for the development of AI systems capable of complex reasoning. In NLP, to better understand human argumentation, argument structure analysis aims to identify argument components, such as claims and premises, and their relations from free text. It encompasses a variety of divergent tasks, such as end-to-end argument mining, argument pair extraction, and argument quadruplet extraction. Existing methods are usually tailored to only one specific argument structure analysis task, overlooking the inherent connections among different tasks. We observe that the fundamental goal of these tasks is similar: identifying argument components and their interrelations. Motivated by this, we present a unified generative framework for argument structure analysis (UniASA). It can uniformly address multiple argument structure analysis tasks in a sequence-to-sequence manner. Further, we enhance UniASA with a multi-view learning strategy based on subtask decomposition. We conduct experiments on seven datasets across three tasks. The results indicate that UniASA can address these tasks uniformly and achieve performance that is either superior to or comparable with the previous state-of-the-art methods. Also, we show that UniASA can be effectively integrated with large language models, such as Llama, through fine-tuning or in-context learning.

## Installation

```bash
conda create -n asa python=3.9.15

conda activate asa

pip install -r requirements.txt
```

## Datasets Preparation

### AAEC

1. Download [ArgumentAnnotatedEssays-2.0.zip](https://tudatalib.ulb.tu-darmstadt.de/handle/tudatalib/2422) and unzip it to `input/orig_datasets`, then unzip `brat-project-final.zip`.

2. Process the ArgumentAnnotatedEssays-2.0 dataset as detailed in `input/orig_datasets/fix_errors_in_AAEC.md`, save the processed dataset as `input/orig_datasets/ArgumentAnnotatedEssays-2.0-processed`. Here, we fix some annotation errors in the original dataset.

### CDCP, AbstRCT, MTC, AASD

1. Process these 4 datasets according to the instructions from [graph_parser](https://github.com/hitachi-nlp/graph_parser/tree/main/examples/multitask_am), and store the processed datasets in `input/orig_datasets` as `cdcp-from_tacl22`, `abstrct-from_tacl22`, `aasd-from_tacl22`, and `mtc-from_tacl22` respectively.

2. Clone this [repo](https://github.com/peldszus/arg-microtexts.git) into `input/orig_datasets` for the topic informatioin os the MTC dataset.

### RR

1. Download fils in [RR-submission-v2](https://github.com/LiyingCheng95/ArgumentPairExtraction/tree/master/data/RR-submission-v2) and save them to `input/orig_datasets/RR-submission-v2`.

### QAM

1. Download fils in [QAM](https://github.com/guojiapub/QuadTAG/tree/main/data/QAM) and save them to `input/orig_datasets/QAM`.

## Running Scripts

1. Run this command for data pre-processing:

    ```bash
    bash running_scripts/run_data_process.sh
    ```

2. Run the following commands to train models for each dataset:

    ```bash
    bash running_scripts/run_aaec.sh

    bash running_scripts/run_abstrct.sh

    bash running_scripts/run_cdcp.sh

    bash running_scripts/run_mtc.sh

    bash running_scripts/run_aasd.sh

    bash running_scripts/run_qam.sh

    bash running_scripts/run_rr.sh
    ```

3. The evaluation results can be found in`output/{dataset_name}/seed_{seed}/predict_results.json`.

## Acknowledgement

- Thanks to [graph_parser](https://github.com/hitachi-nlp/graph_parser/tree/main/examples/multitask_am) for providing the data preprocessing code.
- Thanks to [Huggingface Transformers](https://github.com/huggingface/transformers) library, the code in this work is mainly developed based on this library.

## Citation

If you find our work useful, please cite our paper:

```
@article{10.1162/coli_a_00553,
    author = {Bao, Jianzhu and Jing, Mohan and Dong, Kuicai and Sun, Aixin and Sun, Yang and Xu, Ruifeng},
    title = {UniASA: A Unified Generative Framework for Argument Structure Analysis},
    journal = {Computational Linguistics},
    pages = {1-47},
    year = {2025},
    month = {02},
    issn = {0891-2017},
    doi = {10.1162/coli_a_00553}
}
```