# Grammatical Error Correction:
Systems tested on QALB-2014 and QALB-2015-L1 are trained only on QALB-2014 training data, whereas systems tested on ZAEBUC are trained on QALB-2014 and ZAEBUC (upsampled tenfold). For MADAR CODA, we train only on the MADAR CODA training set. The data used to fine-tune all text editings models is [here]().


## Text Editing Models:
We provide [scripts](scripts) to reproduce the text editing models we built by fine-tuning various BERT models (e.g., AraBERTv02). It is important to note that you need to specify the correct training file corresponding to each experiment. We provide a detailed description of the data we used to build our text editings models in the data directory [here]() (make sure to download the data). Fine-tuning the text editing models is typically done by running a variant of the following scripts:

```bash
DATA_DIR=/path/to/data/dir
BERT_MODEL=/path/to/BERT_model # Or hugging face model id
OUTPUT_DIR=/path/to/output
BATCH_SIZE=32
NUM_EPOCHS=10 # 15 for qalb14+zaebuc_x10
SAVE_STEPS=500
SEED=42

python tag.py \
    --tokenized_data_path $DATA_DIR/train.txt \
    --optim adamw_torch \
    --labels $DATA_DIR/labels.txt \
    --model_name_or_path $BERT_MODEL \
    --input_unit subword-level \ # or word-level
    --output_dir $OUTPUT_DIR \
    --num_train_epochs $NUM_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --save_steps $SAVE_STEPS \
    --seed $SEED \
    --do_train \
    --report_to "none" \
    --overwrite_output_dir
```

We also provide the [scripts](scripts) we used during inference to generate grammatically correct outputs using the fine-tuned models:
1. [scripts/run_tagger_test_checkpoint_iter_arr.sh](scripts/run_tagger_test_checkpoint_iter_arr.sh): runs iterative decoding over all checkpoints
2. [scripts/run_tagger_test_iter.sh](scripts/run_tagger_test_iter.sh): runs iterative decoding on a single checkpoint.
3. [scripts/run_tagger_test.sh](scripts/run_tagger_test.sh): runs inference (without iterative decoding) over a single checkpoint.
4. [scripts/run_tagger_test_checkpoint_iter_pnx_sep.sh](scripts/run_tagger_test_checkpoint_iter_pnx_sep.sh): runs iterative decoding over a single non-punctuation (NoPnx) checkpoint followed by a single iteration of correction using a punctuation (Pnx) checkpoint.

Each of the above scripts will have a variant of the following code based on the experiment we'd like to run:
```bash
model=/path/to/model
test_file=/path/to/text/file
test_file_raw=/path/to/text/raw/file
labels=/path/to/labels

BATCH_SIZE=64
SEED=42
pred_mode=test # or dev

python tag.py \
    --tokenized_data_path $test_file \
    --tokenized_raw_data_path $test_file_raw \
    --labels $labels \
    --input_unit subword-level \
    --task da-gec \
    --model_name_or_path $checkpoint \
    --output_dir $checkpoint \
    --per_device_eval_batch_size $BATCH_SIZE \
    --seed $SEED \
    --do_pred \
    --label_pred_output_file ${pred_mode}.preds.txt \
    --rewrite_pred_output_file ${pred_mode}.txt
```

The generated outputs of our models on the dev and test sets are in the [predictions](predictions) directory.


## LLMs:
We provide the code we used to prompt [gpt-3.5-turbo](llms/run_gpt.py), [gpt-4o](llms/run_gpt.py), [Fanar](llms/run_fanar.py), and [Jais-13B-Chat](llms/run_jais.py) along the full set of prompts. We use the following [scripts](llms/scripts) to prompt the LLMs.
Make sure to provide your OpenAI [API_KEY](https://github.com/balhafni/text-editing/blob/85544acc6f34193d596809cc2aac04b1a596af6f/gec/llms/run_gpt.py#L7) and Fanar [API_KEY](https://github.com/balhafni/text-editing/blob/85544acc6f34193d596809cc2aac04b1a596af6f/gec/llms/run_fanar.py#L8) before prompting the gpt-* and Fanar.


## Ensemble Models:
The code of our majority vote edit ensemble is available in [ensemble.py](ensemble.py). Obtaining the outputs of all of our ensembles can be done by running [scripts/run_ensemble.sh](scripts/run_ensemble.sh).


## Evaluation:
We pick the best model checkpoint based on the development sets. However, the M<sup>2</sup> scorer suffers from extreme running times in cases where the generated outputs differ significantly from the input. To mitigate this bottleneck, we use an extended version of the [M<sup>2</sup> scorer](https://github.com/balhafni/text-editing/tree/master/gec/utils/m2scorer) which was introduce by [Alhafni et al., 2023](https://aclanthology.org/2023.emnlp-main.396/). This version of the M<sup>2</sup> scorer is in python3 and it has a [time limit](https://github.com/balhafni/text-editing/blob/85544acc6f34193d596809cc2aac04b1a596af6f/gec/utils/m2scorer/m2scorer.py#L141) for each sentence during evaluation. If the evaluation of a single generated sentence surpasses this limit, we pass the input sentence to the output without modifications.

This extension is modular and allows us to import and use M<sup>2</sup> scorer to evaluate the generated outputs within other scripts. In cases where the generated sentences are replaced by the input, the generated output file will end with a `.pp` extension. Note that our version of the  M<sup>2</sup> scorer has been tested and is identical to the M<sup>2</sup> scorer released as part of the QALB shared task in terms of evaluation.

During the evaluation, we use the manually created m2edits for QALB-2014 and QALB-2015-L1 which are publicly available as part of the shared task. For ZAEBUC and MADAR CODA, we rely on the m2edits created by [Alhafni et al., 2023](https://aclanthology.org/2023.emnlp-main.396/) and [Alhafni et al., 2024](https://aclanthology.org/2024.arabicnlp-1.4/), respectively. 

Running the development sets evaluation over all checkpoints can be done using the [scripts/run_m2scorer.sh](scripts/run_m2scorer.sh) script. Evaluating a single prediction file can be done using [scripts/run_m2scorer_file.sh](scripts/run_m2scorer_file.sh). The edits coverage oracle experiments are evaluated using [scripts/run_m2scorer_oracle.sh](scripts/run_m2scorer_oracle.sh).
