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
All the datasets we used throughout the paper to train and test various systems can be downloded from [here]().

This repo is organized as follows:
1. [edits](edits): includes the scripts needed to extract edits from parallel GEC corpora and to create different edit representation.
2. [gec](gec): includes the scripts needed to train and evaluate our text editing GEC systems.

## Hugging Face Integration:
We make our text editing models publicly available on [Hugging Face](https://huggingface.co/collections/CAMeL-Lab/text-editing-683efe7eb17f9390d191ae1f).

```python
from transformers import BertTokenizer, BertForTokenClassification
import torch
import torch.nn.functional as F
from gec.tag import rewrite


nopnx_tokenizer = BertTokenizer.from_pretrained('CAMeL-Lab/text-editing-qalb14-nopnx')
nopnx_model = BertForTokenClassification.from_pretrained('CAMeL-Lab/text-editing-qalb14-nopnx')

pnx_tokenizer = BertTokenizer.from_pretrained('CAMeL-Lab/text-editing-qalb14-pnx')
pnx_model = BertForTokenClassification.from_pretrained('CAMeL-Lab/text-editing-qalb14-pnx')


def predict(model, tokenizer, text, decode_iter=1):
    for _ in range(decode_iter):
        tokenized_text = tokenizer(text, return_tensors="pt", is_split_into_words=True)
        with torch.no_grad():
            logits = model(**tokenized_text).logits
            preds = F.softmax(logits.squeeze(), dim=-1)
            preds = torch.argmax(preds, dim=-1).cpu().numpy()
            edits = [model.config.id2label[p] for p in preds[1:-1]]
            assert len(edits) == len(tokenized_text['input_ids'][0][1:-1])
        subwords = tokenizer.convert_ids_to_tokens(tokenized_text['input_ids'][0][1:-1])
        text = rewrite(subwords=[subwords], edits=[edits])[0][0]
    return text


text = 'يجب الإهتمام ب الصحه و لا سيما ف ي الصحه النفسيه ياشباب المستقبل،،'.split()

output_sent = predict(nopnx_model, nopnx_tokenizer, text, decode_iter=2)
output_sent = predict(pnx_model, pnx_tokenizer, output_sent.split(), decode_iter=1)
print(output_sent) # يجب الاهتمام بالصحة ولا سيما في الصحة النفسية يا شباب المستقبل .
```


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
