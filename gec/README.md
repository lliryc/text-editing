# Grammatical Error Correction:
Systems tested on QALB-2014 and QALB-2015-L1 are trained only on QALB-2014 training data, whereas systems tested on ZAEBUC are trained on QALB-2014 and ZAEBUC (upsampled tenfold). For MADAR CODA, we train only on the MADAR CODA training set. The data used to fine-tune all text editings models is [here]().


## Text Editing Models:


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
