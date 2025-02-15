import json
from alignment.aligner import word_level_alignment, char_level_alignment
from edit import Edit, SubwordEdits, SubwordEdit
from utils import (apply_edits, insert_to_append)
import re
import argparse
import os
from collections import Counter
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from util.postprocess import remove_pnx, pnx_tokenize, space_clean

def read_data(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f.readlines()]


def read_data_txt(path):
    with open(path) as f:
        return [line.strip() for line in f.readlines()]


def create_edits(char_level_alignment, word_level_alignment):
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

        _word_edits = SubwordEdits.create(src_words, word_edit.edit)
        word_edits.append(_word_edits)

        _subword_edits = SubwordEdits.create(src_words, word_edit.edit)
        subword_edits.append(_subword_edits)

    assert len(subword_edits) == len(word_edits) == len(aligned_src_chars) == len(aligned_tgt_chars)

    try:
        # flatten the subword edits
        flatten_subword_edits = [edit for subword_edit in subword_edits for edit in subword_edit.edits]
        # converting the insertions to appends at the subword-level
        flatten_subword_edits_w_appends = insert_to_append(flatten_subword_edits)

        # flatten the word edits
        flatten_word_edits = [edit for word_edit in word_edits for edit in word_edit.edits]
        # converting the insertions to appends at the word-level
        word_edits_w_appends = insert_to_append(flatten_word_edits)

    except:
        import pdb; pdb.set_trace()

    return {'word-edits': word_edits, 'word-edits-append': word_edits_w_appends}


def create_dataset_edits(dataset, direction='raw-cor'):
    dataset_w_edits = []

    for i, example in enumerate(dataset):
        if i % 1000 == 0:
            print(i, flush=True)
        src_sent = example['raw'] if direction == 'raw-cor' else example['cor']
        tgt_sent = example['cor'] if direction == 'raw-cor' else example['raw']

        word_level_align = word_level_alignment(src_sent=src_sent,
                                                tgt_sent=tgt_sent)

        char_level_align = char_level_alignment(word_level_align)

        example_edits = create_edits(char_level_align, word_level_align)

        word_edits = example_edits['word-edits']
        word_edits_append = example_edits['word-edits-append']

        rewritten_src_word_edits = apply_edits(src_sent.split(), word_edits_append)

        if ' '.join(rewritten_src_word_edits) != tgt_sent:
            import pdb; pdb.set_trace()

        dataset_w_edits.append({'src': src_sent, 'tgt': tgt_sent,
                                'word-level-align': word_level_align,
                                'char-level-align': char_level_align,
                                'word-edits': word_edits,
                                'word-edits-append': word_edits_append})

    return dataset_w_edits


def process_example(example, direction):
    src_sent = example['raw'] if direction == 'raw-cor' else example['cor']
    tgt_sent = example['cor'] if direction == 'raw-cor' else example['raw']

    word_level_align = word_level_alignment(src_sent=src_sent, tgt_sent=tgt_sent)
    char_level_align = char_level_alignment(word_level_align)

    example_edits = create_edits(char_level_align, word_level_align)
    word_edits = example_edits['word-edits']
    word_edits_append = example_edits['word-edits-append']

    rewritten_src_word_edits = apply_edits(src_sent.split(), word_edits_append)

    if ' '.join(rewritten_src_word_edits) != tgt_sent:
        import pdb; pdb.set_trace()

    return {
        'src': src_sent,
        'tgt': tgt_sent,
        'word-level-align': word_level_align,
        'char-level-align': char_level_align,
        'word-edits': word_edits,
        'word-edits-append': word_edits_append
    }


def create_dataset_edits_parallel(dataset, direction='raw-cor', num_workers=4):
    dataset_w_edits = [None] * len(dataset)  # Preallocate space for results


    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_example, example, direction): idx
            for idx, example in enumerate(dataset)
        }
        processed_examples = 0
        for future in as_completed(futures):
            try:
                idx = futures[future]  # Get the index of the original example
                dataset_w_edits[idx] = future.result()
                processed_examples += 1

                if processed_examples % 100 == 0:
                    print(f"Processed {processed_examples} examples...", flush=True)

            except Exception as e:
                print(f"Error processing example at index {idx}: {e}")

    return dataset_w_edits


def ensemble_rewrite(models_edits, voting_threshold=2):
    # Initialize examples with subwords from the first model. We can do this
    # because all the models have the same src (i.e., subwords)
    examples = [
        {(edit.subword, idx): Counter() for idx, edit in enumerate(subword_edits['word-edits-append'])}
        for subword_edits in models_edits[0]
    ]
    
    # Aggregate edits from all models
    for i, edits_group in enumerate(zip(*models_edits)):
        for model_edits in edits_group:
            for idx, edit in enumerate(model_edits['word-edits-append']):
                examples[i][(edit.subword, idx)][edit.edit] += 1

    # Keeping the edits that appear more than voting_threshhold time!
    _examples = []
    for i, example in enumerate(examples):
        _example = []

        for j, subword in enumerate(example):
            subword_edits_cnt = example[subword]
            edit = [k for k, v in subword_edits_cnt.items() if v >= voting_threshold]
            if edit:
                edit = edit[0]
                if edit == 'D':
                    edit = 'D*'
                _example.append(SubwordEdit(subword[0], subword[0], edit))
            else:
                # import pdb; pdb.set_trace()
                # trust the text-editing model
                # _example.append(SubwordEdit(subword[0], models_edits[0][i]['subword-edits-append'][j].edit)) 
                # _example.append(SubwordEdit(subword[0], 'K'))
                _example.append(SubwordEdit(subword[0], subword[0], 'K'))
        
        _examples.append(_example)

    detok_rewritten_sents, rewritten_sents, non_app_edits = rewrite(_examples)
    return detok_rewritten_sents


def rewrite(examples):

    rewritten_sents = []
    rewritten_sents_merge = []
    non_app_edits = []

    for example in examples:
        sent_subwords = [e.subword for e in example]
        sent_edits = [e.edit for e in example]

        assert len(sent_subwords) == len(sent_edits)
        rewritten_sent = []

        for subword, edit in zip(sent_subwords, sent_edits):
            edit = SubwordEdit(subword, subword, edit)

            if edit.is_applicable(subword):
                rewritten_subword = edit.apply(subword)
                rewritten_sent.append(rewritten_subword)
            else:
                non_app_edits.append({'subword': subword, 'edit': edit.to_json_str()})
                rewritten_sent.append(subword)

        rewritten_sents_merge.append(resolve_merges(rewritten_sent, sent_edits))
        rewritten_sents.append(rewritten_sent)


    rewritten_sents_merge = [' '.join(sent) for sent in rewritten_sents_merge]
    return rewritten_sents_merge, rewritten_sents, non_app_edits


def resolve_merges(sent, edits):
    _sent = []
    assert len(sent) == len(edits)
    for subword, edit in zip(sent, edits):
        if edit.startswith('M'):
            if len(_sent) > 0:
                _sent[-1] = _sent[-1] + subword
            else:
                _sent.append(subword)
        else:
            _sent.append(subword)
    return _sent



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', default=None)
    parser.add_argument('--models_outputs', nargs='+', default=None)
    parser.add_argument('--voting_threshold', default=2, type=int)
    parser.add_argument('--pnx_proc', action='store_true')
    parser.add_argument('--output_path')

    
    args = parser.parse_args()
    input_data = read_data_txt(args.input_file)

    models_outputs = [read_data_txt(path) for path in args.models_outputs]
    models_edits = []


    for model_output in models_outputs:
        assert len(model_output) == len(input_data)
        data = [{'raw': input_data[i], 'cor': model_output[i]} for i in range(len(input_data))]
        edits_data = create_dataset_edits_parallel(data, direction='raw-cor', num_workers=100)
        # edits_data = create_dataset_edits(data, direction='raw-cor')
        models_edits.append(edits_data)

    rewritten_examples = ensemble_rewrite(models_edits, voting_threshold=args.voting_threshold)

    if args.pnx_proc:
        rewritten_examples = pnx_tokenize(rewritten_examples)
        rewritten_examples_nopnx = remove_pnx(rewritten_examples)
    else:
        rewritten_examples = space_clean(rewritten_examples)


    with open(args.output_path, mode='w') as f:
        for example in rewritten_examples:
            f.write(example)
            f.write('\n')

    if args.pnx_proc:
        with open(f'{args.output_path}.nopnx', mode='w') as f:
            for example in rewritten_examples_nopnx:
                f.write(example)
                f.write('\n')