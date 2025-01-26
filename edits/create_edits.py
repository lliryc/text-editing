import json
from tokenizer import Tokenizer
from alignment.aligner import word_level_alignment, char_level_alignment
from edit import Edit, SubwordEdits, SubwordEdit
from utils import insert_to_append, write_json, load_data, write_tsv, get_stats, write_cooccur
import re
import argparse


def read_data(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f.readlines()]


def create_edits(char_level_alignment, word_level_alignment, tokenizer):
    """
    Args:
        char_level_alignment (dict): pair of sentences aligned at the
            char level
        word_level_alignment (dict): pair of sentences aligned at the
            word level

    Returns:
        dict of word-level edits and subword-level edits:
            the word-level edits are in a list of Edit
            the subword-level edits are in a list of SubwordEdits
    """
    aligned_src_words = word_level_alignment['src']
    aligned_tgt_words = word_level_alignment['tgt']

    aligned_src_chars = char_level_alignment['src']
    aligned_tgt_chars = char_level_alignment['tgt']

    assert (len(aligned_src_words) == len(aligned_tgt_words) ==
            len(aligned_src_chars) == len(aligned_tgt_chars))

    word_edits = []
    subword_edits = []

    for i in range(len(aligned_src_words)):
        src_word_chars = aligned_src_chars[i]
        tgt_word_chars = aligned_tgt_chars[i]

        src_words = aligned_src_words[i]
        tgt_words = aligned_tgt_words[i]

        assert len(src_word_chars) == len(tgt_word_chars)
        word_edit = Edit.create(src_word_chars, tgt_word_chars)
        word_edits.append(word_edit)

        _subword_edits = SubwordEdits.create(src_words, word_edit.edit, tokenizer)
        subword_edits.append(_subword_edits)

    assert len(subword_edits) == len(word_edits) == len(aligned_src_chars) == len(aligned_tgt_chars)

    try:
        # flatten the subword edits
        flatten_subword_edits = [edit for subword_edit in subword_edits for edit in subword_edit.edits]

        # converting the insertions to appends
        flatten_edits_w_appends = insert_to_append(flatten_subword_edits)
    except:
        import pdb; pdb.set_trace()

    return {'word-edits': word_edits, 'subword-edits': flatten_subword_edits,
            'subword-edits-append': flatten_edits_w_appends}


def create_dataset_edits(dataset, tokenizer, direction='raw-cor'):
    dataset_w_edits = []

    for i, example in enumerate(dataset):
        src_sent = example['raw'] if direction == 'raw-cor' else example['cor']
        tgt_sent = example['cor'] if direction == 'raw-cor' else example['raw']

        word_level_align = word_level_alignment(src_sent=src_sent,
                                                tgt_sent=tgt_sent)
        
        char_level_align = char_level_alignment(word_level_align)

        example_edits = create_edits(char_level_align, word_level_align, tokenizer)

        word_edits = example_edits['word-edits']
        subword_edits = example_edits['subword-edits']
        subword_edits_append = example_edits['subword-edits-append']

        tokenized_src = tokenizer.tokenize(src_sent, flatten=True)
        rewritten_src = apply_edits(tokenized_src, subword_edits_append)

        if ' '.join(rewritten_src) != tgt_sent:
            import pdb; pdb.set_trace()

        dataset_w_edits.append({'src': src_sent, 'tgt': tgt_sent,
                                'word-level-align': word_level_align,
                                'char-level-align': char_level_align,
                                'word-edits': word_edits,
                                'subword-edits': subword_edits,
                                'subword-edits-append': subword_edits_append})

    return dataset_w_edits


def apply_edits(tokenized_text, edits):
    assert len(tokenized_text) == len(edits)

    rewritten_txt = []

    for subword, edit in zip(tokenized_text, edits):
        rewritten_subword = edit.apply(subword)

        edit_ops = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|A_\[.*?\]+|D+|K+|.', edit.edit)

        if 'M' in edit_ops: # merge
            rewritten_txt[-1] = rewritten_txt[-1] + rewritten_subword
        else:
            rewritten_txt.append(rewritten_subword)

    # collapsing subwords
    _rewritten_txt = []
    for subword in rewritten_txt:
        if subword.startswith('##'):
            _rewritten_txt[-1] = _rewritten_txt[-1] + subword.replace('##','')
        else:
            _rewritten_txt.append(subword)

    # take out complete deletions
    _rewritten_txt = [subword.strip() for subword in _rewritten_txt if subword != '']
    return _rewritten_txt


if __name__ == '__main__':
    # for split in ['train', 'dev', 'test']:
    #     print(split)
    #     data = read_data(f'../arabic-gec/data/gec/modeling/zaebuc/wo_camelira/full/{split}.json')
    #     tokenizer = Tokenizer('CAMeL-Lab/bert-base-arabic-camelbert-msa')

        # edits_data = create_dataset_edits(data, tokenizer, direction='raw-cor')
        # write_json(path=f'edits_outputs/zaebuc/{split}_edits.json', data=edits_data)
        # write_tsv(path=f'edits_outputs/zaebuc/{split}_edits', data=edits_data)

        # error_injection_edits = create_dataset_edits(data, tokenizer, direction='cor-raw')
        # write_json(path=f'{split}_edits.jsonbla', data=error_injection_edits)
        # write_tsv(path=f'{split}_editsbla', data=error_injection_edits)

    data = load_data(f'edits_outputs/qalb14/train_edits.json')
    get_stats(data, 'edits_outputs/qalb14/train')
    # write_tsv(path='qalb14_train_edits', data=data)


    # write_cooccur(data, 'qalb14_train_cnts.txt')
    # edit = Edit.create(['ا', 'ل', 'ش', 'ي', 'ع', 'ة ،', ' ', 'ا', 'ل', 'س', 'ن', 'ة'],
    #                    ['ا', 'ل', 'ش', 'ي', 'ع', 'ه_', '', 'ا', 'ل', 'س', 'ن', 'ه'])

    # edits = SubwordEdits.create('من دعم', 'DDI_[ب]MKKK', tokenizer)
    # import pdb; pdb.set_trace()
    # edit = SubwordEdit(subword='فيه', edit='DK*')
    # edit.apply('فيه')

