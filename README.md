# Enhancing Text Editing for Grammatical Error Correction

This repo contains code and pretrained models to reproduce the results in our paper [Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study](https://arxiv.org/abs/2503.00985).

## Requirements:

The code was written for python>=3.10, pytorch 1.12.1, and transformers 4.30.0. You will need a few additional packages. Here's how you can set up the environment using conda (assuming you have conda and cuda installed):

```bash

git clone https://github.com/CAMeL-Lab/text-editing.git
cd text-editing

conda create -n text-editing python=3.10
conda activate text-editing

pip install -e .
```

## Experiments and Reproducibility:

This repo is organized as follows:

1. [data](data): includes all the data we used throughout our paper to train and test various systems.
2. [edits](edits): includes the scripts needed to extract edits from parallel GEC corpora and to create different edit representation.
3. [gec](gec): includes the scripts needed to train and evaluate our text editing GEC systems.


## License:
This repo is available under the MIT license. See the [LICENSE](LICENSE) for more info.


## Citation:
If you find the code or data in this repo helpful, please cite our [paper](https://arxiv.org/abs/2503.00985):

```bibtex
@misc{alhafni-habash-2025-enhancing,
      title={Enhancing Text Editing for Grammatical Error Correction: Arabic as a Case Study}, 
      author={Bashar Alhafni and Nizar Habash},
      year={2025},
      eprint={2503.00985},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2503.00985}, 
}
```
