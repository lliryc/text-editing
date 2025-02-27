import json
from tokenizer import Tokenizer
from alignment.aligner import word_level_alignment, char_level_alignment
from edit import Edit, SubwordEdits
from utils import (apply_edits, insert_to_append, compress_edits, write_json,
                   load_data, write_tsv, get_stats, prune_edits, prune_edits_corr)
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


def read_data(path):
    with open(path) as f:
        return [json.loads(line.strip()) for line in f.readlines()]


def read_data_txt(src_path, tgt_path):
    examples = []
    with open(src_path) as src_f, open(tgt_path) as tgt_f:
        for src, tgt in zip(src_f.readlines(), tgt_f.readlines()):
            examples.append({'raw': src.strip(), 'cor': tgt.strip()})
    return examples



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

        _word_edits = SubwordEdits.create(src_words, word_edit.edit)
        word_edits.append(_word_edits)

        _subword_edits = SubwordEdits.create(src_words, word_edit.edit, tokenizer)
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

    return {'word-edits': word_edits, 'word-edits-append': word_edits_w_appends,
            'subword-edits': flatten_subword_edits,
            'subword-edits-append': flatten_subword_edits_w_appends}


def create_dataset_edits(dataset, tokenizer, direction='raw-cor'):
    dataset_w_edits = []

    for i, example in enumerate(dataset):
        if i % 1000 == 0:
            print(i, flush=True)
        src_sent = example['raw'] if direction == 'raw-cor' else example['cor']
        tgt_sent = example['cor'] if direction == 'raw-cor' else example['raw']

        word_level_align = word_level_alignment(src_sent=src_sent,
                                                tgt_sent=tgt_sent)

        char_level_align = char_level_alignment(word_level_align)

        example_edits = create_edits(char_level_align, word_level_align, tokenizer)

        word_edits = example_edits['word-edits']
        word_edits_append = example_edits['word-edits-append']
        subword_edits = example_edits['subword-edits']
        subword_edits_append = example_edits['subword-edits-append']

        tokenized_src_raw, tokenized_src_internal = tokenizer.tokenize(src_sent, flatten=True)

        rewritten_src_subword_edits = apply_edits(tokenized_src_raw, subword_edits_append)

        if ' '.join(rewritten_src_subword_edits) != tgt_sent:
            import pdb; pdb.set_trace()

        rewritten_src_word_edits = apply_edits(src_sent.split(), word_edits_append)

        if ' '.join(rewritten_src_word_edits) != tgt_sent:
            import pdb; pdb.set_trace()

        dataset_w_edits.append({'src': src_sent, 'tgt': tgt_sent,
                                'word-level-align': word_level_align,
                                'char-level-align': char_level_align,
                                'word-edits': word_edits,
                                'word-edits-append': word_edits_append,
                                'subword-edits': subword_edits,
                                'subword-edits-append': subword_edits_append})

    return dataset_w_edits



def process_example(example, tokenizer, direction):
    src_sent = example['raw'] if direction == 'raw-cor' else example['cor']
    tgt_sent = example['cor'] if direction == 'raw-cor' else example['raw']

    word_level_align = word_level_alignment(src_sent=src_sent, tgt_sent=tgt_sent)
    char_level_align = char_level_alignment(word_level_align)

    example_edits = create_edits(char_level_align, word_level_align, tokenizer)
    word_edits = example_edits['word-edits']
    word_edits_append = example_edits['word-edits-append']
    subword_edits = example_edits['subword-edits']
    subword_edits_append = example_edits['subword-edits-append']

    tokenized_src_raw, tokenized_src_internal = tokenizer.tokenize(src_sent, flatten=True)

    rewritten_src_subword_edits = apply_edits(tokenized_src_raw, subword_edits_append)

    if ' '.join(rewritten_src_subword_edits) != tgt_sent:
        import pdb; pdb.set_trace()

    rewritten_src_word_edits = apply_edits(src_sent.split(), word_edits_append)

    if ' '.join(rewritten_src_word_edits) != tgt_sent:
        import pdb; pdb.set_trace()

    return {
        'src': src_sent,
        'tgt': tgt_sent,
        'word-level-align': word_level_align,
        'char-level-align': char_level_align,
        'word-edits': word_edits,
        'word-edits-append': word_edits_append,
        'subword-edits': subword_edits,
        'subword-edits-append': subword_edits_append,
    }


def create_dataset_edits_parallel(dataset, tokenizer, direction='raw-cor', num_workers=4):
    dataset_w_edits = [None] * len(dataset)  # Preallocate space for results


    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_example, example, tokenizer, direction): idx
            for idx, example in enumerate(dataset)
        }
        processed_examples = 0
        for future in as_completed(futures):
            try:
                idx = futures[future]  # Get the index of the original example
                dataset_w_edits[idx] = future.result()
                processed_examples += 1

                if processed_examples % 1000 == 0:
                    print(f"Processed {processed_examples} examples...", flush=True)

            except Exception as e:
                print(f"Error processing example at index {idx}: {e}")

    return dataset_w_edits


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default='train')
    parser.add_argument('--dataset', default='qalb14')
    parser.add_argument('--tokenizer', default=None, required=True)
    parser.add_argument('--create_edits', action='store_true')
    parser.add_argument('--edits_granularity', default='subword')
    parser.add_argument('--src_file_path', default=None)
    parser.add_argument('--tgt_file_path', default=None)
    parser.add_argument('--output_data_dir', default=None)
    parser.add_argument('--compress', action='store_true')
    parser.add_argument('--compress_output_dir', default=None)
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--prune_cor', action='store_true')
    parser.add_argument('--pruned_output_dir', default=None)
    parser.add_argument('--k', default=0, type=int)


    args = parser.parse_args()
    split = args.split

    print(split, flush=True)


    tokenizer = Tokenizer(args.tokenizer)

    output_dir = args.dataset
    output_dir += ('/subword-level' if args.edits_granularity == 'subword'
                    else  f'/word-level')

    if args.create_edits:
        output_data_dir = os.path.join(args.output_data_dir, output_dir)

        if not os.path.exists(output_data_dir):
            os.makedirs(output_data_dir)

        data = read_data_txt(src_path=args.src_file_path, tgt_path=args.tgt_file_path)

        # edits_data = create_dataset_edits(data, tokenizer, direction='raw-cor')
        edits_data = create_dataset_edits_parallel(data, tokenizer, direction='raw-cor', num_workers=100)
        print(f'Done creating edits!', flush=True)
        write_json(path=f'{output_data_dir}/{split}_edits.json', data=edits_data,
                   edits_granularity=args.edits_granularity)

        write_tsv(path=f'{output_data_dir}/{split}', data=edits_data,
                  edits_granularity=args.edits_granularity)

        get_stats(data=edits_data, path=f'{output_data_dir}/{split}',
                  edits_granularity=args.edits_granularity)


    # compressing the data
    if args.compress:
        compress_output_dir = os.path.join(args.compress_output_dir, output_dir)

        if not os.path.exists(compress_output_dir):
            os.makedirs(compress_output_dir)

        if split != 'train':
            test_data = load_data(path=f'{output_data_dir}/{split}_edits.json',
                            edits_granularity=args.edits_granularity)

            # use the compress map of qalb14 when compressing qalb15 L1 test
            if 'qalb15' in args.dataset:
                assert split == 'test'
                compress_map_output = f'{compress_output_dir.replace("qalb15", "qalb14")}/compress_map.json'
            else:
                compress_map_output = f'{compress_output_dir}/compress_map.json'

            compressed_data = compress_edits(test_data=test_data, edits_granularity=args.edits_granularity,
                                             compress_map_output_path=compress_map_output)
        else:
            train_data = load_data(path=f'{output_data_dir}/{split}_edits.json',
                                   edits_granularity=args.edits_granularity)

            compressed_data = compress_edits(train_data=train_data, edits_granularity=args.edits_granularity,
                                             compress_map_output_path=f'{compress_output_dir}/compress_map.json')


        write_json(path=f'{compress_output_dir}/{split}_edits.json', data=compressed_data,
                   edits_granularity=args.edits_granularity)
        write_tsv(path=f'{compress_output_dir}/{split}', data=compressed_data,
                  edits_granularity=args.edits_granularity)

        get_stats(data=compressed_data, path=f'{compress_output_dir}/{split}',
                  edits_granularity=args.edits_granularity)


    if args.prune:
        prune_output_dir = os.path.join(args.pruned_output_dir, output_dir)
        if not os.path.exists(prune_output_dir):
            os.makedirs(prune_output_dir)

        data = load_data(os.path.join(args.compress_output_dir, f'{output_dir}/{split}_edits.json'),
                         edits_granularity=args.edits_granularity)

        if args.prune_cor:
            pruned_data = prune_edits_corr(data, k=args.k)
        else:
            pruned_data = prune_edits(data, k=args.k, edits_granularity=args.edits_granularity)

        write_json(path=f'{prune_output_dir}/{split}_edits.json', data=pruned_data,
                   edits_granularity=args.edits_granularity)
        write_tsv(path=f'{prune_output_dir}/{split}', data=pruned_data,
                  edits_granularity=args.edits_granularity)

        get_stats(data=pruned_data, path=f'{prune_output_dir}/{split}', edits_granularity=args.edits_granularity)
