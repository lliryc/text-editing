# Grammatical Error Correction:
Systems tested on QALB-2014 and QALB-2015-L1 are trained only on QALB-2014 training data, whereas systems tested on ZAEBUC are trained on QALB-2014 and ZAEBUC (upsampled tenfold). For MADAR CODA, we train only on the MADAR CODA training set. The data used to fine-tune all text editings models is [here]().


## LLMs:


## Text Editing Models:


## Ensemble Models:


## Evaluation:

We pick the best model checkpoint based on the development sets. However, the M<sup>2</sup> scorer suffers from extreme running times in cases where the generated outputs differ significantly from the input. To mitigate this bottleneck, we use an extended version of the [M<sup>2</sup> scorer](https://github.com/balhafni/text-editing/tree/master/gec/utils/m2scorer) which was introduce by [Alhafni et al., 2023](). This version of the M<sup>2</sup> scorer is in python3 and it has a [time limit](https://github.com/CAMeL-Lab/arabic-gec/blob/master/gec/utils/m2scorer/m2scorer.py#L141) for each sentence during evaluation. If the evaluation of a single generated sentence surpasses this limit, we pass the input sentence to the output without modifications.

This extension is modular and allows us to import and use M<sup>2</sup> scorer to evaluate the generated outputs within other scripts. In cases where the generated sentences are replaced by the input, the generated output file will end with a `.pp` extension. Note that our version of the  M<sup>2</sup> scorer has been tested and is identical to the M<sup>2</sup> scorer released as part of the QALB shared task in terms of evaluation.

During the evaluation, we use the manually created m2edits for QALB-2014 and QALB-2015-L1 which are publicly available as part of the shared task. For ZAEBUC and MADAR CODA, we rely on the m2edits created by [Alhafni et al., 2023]() and [Alhafni et al., 2024](), respectively. 

Running the development sets evaluation over all checkpoints can be done using the [run_m2scorer.sh](run_m2scorer.sh) script. Evaluating a single prediction file can be done using [run_m2scorer_file.sh](run_m2scorer_file.sh). The edits coverage oracle experiments are evaluated using [run_m2scorer_oracle.sh](run_m2scorer_oracle.sh).
