# Data and Edit Extraction:

We used the [QALB-2014](https://camel.abudhabi.nyu.edu/qalb-shared-task-2015/), [QALB-2015](https://camel.abudhabi.nyu.edu/qalb-shared-task-2015/), [ZAEBUC](https://sites.google.com/view/zaebuc/home), and the [MADAR Coda Corpus](https://camel.abudhabi.nyu.edu/madar-coda-corpus/) datasets to train and evaluate our models.
For the QALB-2014, we use the publicly available train, dev, and test splits. For QALB-2015, we only use the L1 test data.
For ZAEBUC and Madar Coda, we use the splits made available by [Alhafni et al. 2023](https://github.com/CAMeL-Lab/arabic-gec/tree/master/data) and [Alhafni et al. 2024](https://github.com/CAMeL-Lab/codafication/tree/master/data), respectively.

**Note**: We provide all the datasets used for creating the edits at this [link](). Additionally, we share the extracted edits along with their various representations from our best setup (using AraBERTv02). To make sure that everything runs smoothly, download the data and place it at the root directory of this repo.

We describe below how to extract the edits from parallel GEC datasets along with creating their different representations.


## Edit Extraction:

Our edit extraction algorithm is described in the paper. To extract edits from parallel GEC datasets, you should run the following:

```bash
src=/path/to/src/file
tgt=/path/to/tgt/file
dataset_name=dataset_name #e.g., qalb14
tokenizer=/path/to/bert/model # or huggingface model id
gran=word-level # or subword-level
output_dir=/output/dir # non-compressed edits output dir
output_dir_comp=/output_comp/dir # compressed edits output dir

python create_edits.py \
  --dataset ${dataset_name} \
  --tokenizer ${tokenizer} \
  --split train \
  --src_file_path ${src} \
  --tgt_file_path ${tgt} \
  --create_edits \
  --edits_granularity ${gran} \
  --output_data_dir ${output_dir} \
  --compress \
  --compress_output_dir ${output_dir_comp}
```

Pruning edits can be done as follows:

```bash
dataset_name=dataset_name #e.g., qalb14
tokenizer=/path/to/bert/model # or huggingface model id
output_dir_comp=/output_comp/dir # compressed edits output dir
pruned_output_dir=/output_pruned/dir # pruned edits output dir

python create_edits.py \
  --dataset ${dataset_name} \
  --tokenizer ${tokenizer} \
  --split train \
  --edits_granularity subword \
  --compress_output_dir ${output_dir_comp} \
  --prune \
  --k 10 \ # or 20, 30
  --pruned_output_dir ${output_dir_prune}
```

Running `bash scripts/create_edits.sh` extracts all edits from the parallel GEC datasets we train and report on.
Specifically, it creates non-compressed and compressed edits at both the word and subword levels along with the pruned edits.
Note that pruning is only applied to the training data.


Once the edits have been extracted, we segregate punctuation from non-punctuation edits can be done by running:
```bash
dataset_name=dataset_name #e.g., qalb14
tokenizer=/path/to/bert/model # or huggingface model id
input_data_dir=/path/to/non-compressed/edits
output_data_dir=/path/to/output/dir

python create_edits_pnx_sep.py \
  --dataset ${dataset_name} \
  --tokenizer ${tokenizer} \
  --split train \
  --create_edits \
  --create_pnx_edits \
  --edits_granularity subword \
  --input_data_dir ${input_data_dir} \
  --output_data_dir ${output_data_dir}
```


Running `bash scripts/create_edits_pnx_sep.sh` will segregate the punctuation edits from the MSA GEC datasets we report on. It will also create the the pruned punctuation and non-punctuation edits for the training data.

In the edit extraction scripts we provide ([create_edits.sh](scripts/create_edits.sh) and [create_edits_pnx_sep.sh](scripts/create_edits_pnx_sep.sh)), we use the tokenizer of AraBERTv02. Replacing this tokenizer with the tokenizer from CAMeLBERT-MSA or ARBERTv2 will get you the data you need to replicate the rest of the experiments we report on in the paper.


## Data:


## Tokenizer:
