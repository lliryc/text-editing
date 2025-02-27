# Data and Edit Extraction:

We used the [QALB-2014](https://camel.abudhabi.nyu.edu/qalb-shared-task-2015/), [QALB-2015](https://camel.abudhabi.nyu.edu/qalb-shared-task-2015/), [ZAEBUC](https://sites.google.com/view/zaebuc/home), and the [MADAR Coda Corpus](https://camel.abudhabi.nyu.edu/madar-coda-corpus/) datasets to train and evaluate our models.
For the QALB-2014, we use the publicly available train, dev, and test splits. For QALB-2015, we only use the L1 test data.
For ZAEBUC and Madar Coda, we use the splits made available by [Alhafni et al. 2023](https://github.com/CAMeL-Lab/arabic-gec/tree/master/data) and [Alhafni et al. 2024](https://github.com/CAMeL-Lab/codafication/tree/master/data), respectively.

**Note**: We provide all the datasets used for creating the edits at this [link](). Additionally, we share the extracted edits along with their various representations from our best setup (using AraBERTv02).

We describe below how to extract the edits from parallel GEC datasets along with creating their different representations.


## Edit Extraction:

Our edit extraction algorithm is described in the paper. Running `bash scripts/create_edits.sh` extracts all edits from the parallel GEC datasets we train and report on.
Specifically, it creates non-compressed and compressed edits at both the word and subword levels along with the pruned edits.
Note that pruning is only applied to the training data.

Once the edits have been extracted, you can segregate punctuation from non-punctuation edits by running `bash scripts/create_edits_pnx_sep.sh`. This will also create the the pruned punctuation and non-punctuation edits for the training data.

In the edit extraction scripts we provide, we use the tokenizer of AraBERTv02. Replacing this tokenizer with the tokenizer from CAMeLBERT-MSA or ARBERTv2 will get you the data you need to replicate the rest of the experiments we report on in the paper.


## Data:

