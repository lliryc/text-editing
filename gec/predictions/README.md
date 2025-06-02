# Grammatical Error Correction Outputs:

We provide the outputs on the dev and test sets of QALB-2014, QALB-2015-L1, ZAEBUC, and MADAR Coda of all the experiments we report on:
1. `oracle`: outputs of the oracle experiments we report on as part of edits coverage.
2. `seq2seq`: outputs of the models developed by [Alhafni et al., 2023](https://aclanthology.org/2023.emnlp-main.396/) and [Alhafni et al., 2024](https://aclanthology.org/2024.arabicnlp-1.4/).
3. `llms`: outputs of the LLMs we report on.
4. `taggers`: outputs of all the text editing experiments we report on using AraBERTv02, ARBERTv2, and CAMeLBERT-MSA
5. `ensembles`: outputs of the ensemble models.

Each of the subfolders in each directory could potentially have files with the following extensions:

1. `*.txt`: the models' outputs  on the development/test sets. If there's a digit (e.g., `dev.1.txt`), this indicates the correction iteration if applicable.
2. `*.txt.pp`: the models' outputs  of the development/test sets after replacing the generated sentences that differ significantly from the input by the input sentences.
3. `*.m2`: the m2scorer evaluation of the `*.txt` files.
5. `*.nopnx.txt`: the models' no-punctuation outputs on the development/test sets.
6. `[qalb14|qalb15|zaebuc]_test.preds.txt.m2]`: the m2scorer evaluation of the `[qalb14|qalb15|zaebuc]_test.preds.txt]` files
7. `*.nopnx.txt`: the m2scorer evaluation of the `*.nopnx.txt` files
