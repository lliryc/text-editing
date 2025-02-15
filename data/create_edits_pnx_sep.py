import json
from tokenizer import Tokenizer
from alignment.aligner import word_level_alignment, char_level_alignment
from edit import Edit, SubwordEdits, SubwordEdit
from utils import (apply_edits, insert_to_append, compress_edits, write_json,
                   load_data, separate_pnx_edits, write_tsv, get_stats, prune_edits, prune_edits_corr)
import re
import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed


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


def create_dataset_edits(dataset, tokenizer):
    dataset_w_edits = []

    for i, example in enumerate(dataset):
        if i % 1000 == 0:
            print(i)

        src_sent = example['cor-no-pnx']
        tgt_sent = example['tgt']

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


def process_example(example, tokenizer):
    src_sent = example['cor-no-pnx']
    tgt_sent = example['tgt']

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


def create_dataset_edits_parallel(dataset, tokenizer, num_workers=4):
    dataset_w_edits = [None] * len(dataset)  # Preallocate space for results


    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {
            executor.submit(process_example, example, tokenizer): idx
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



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--split', default='train')
    parser.add_argument('--create_edits', action='store_true')
    parser.add_argument('--tokenizer', default=None, required=True)
    parser.add_argument('--edits_granularity', default='subword')
    parser.add_argument('--create_pnx_edits', action='store_true')
    parser.add_argument('--input_data_dir', default=None)
    parser.add_argument('--output_data_dir', default=None)
    parser.add_argument('--token', default=None)
    parser.add_argument('--prune', action='store_true')
    parser.add_argument('--prune_cor', action='store_true')
    parser.add_argument('--pruned_output_dir', default=None)
    parser.add_argument('--k', default=0, type=int)


    args = parser.parse_args()
    split = args.split

    print(split)

    # if args.token == '5':
    #     tokenizer = Tokenizer('tokenizer-5')
    # elif args.token == '4':
    #     tokenizer = Tokenizer('tokenizer-4')
    # elif args.token == '3':
    #     tokenizer = Tokenizer('tokenizer-3')
    # else:
        # tokenizer = tokenizer = Tokenizer('/scratch/ba63/BERT_models/bert-base-arabertv02')
        # tokenizer = tokenizer = Tokenizer('/scratch/ba63/BERT_models/ARBERTv2')
        # tokenizer = Tokenizer('/scratch/ba63/BERT_models/bert-base-arabic-camelbert-msa')
        # tokenizer = Tokenizer('/scratch/ba63/BERT_models/bert-base-arabic-camelbert-mix')

    tokenizer = Tokenizer(args.tokenizer)

    output_dir = f'{args.dataset}_{args.token}' if args.token else args.dataset
    output_dir += ('/subword-level-check' if args.edits_granularity == 'subword'
                   else  f'/word-level')

    if args.create_edits:
        # loading the data
        data = load_data(os.path.join(args.input_data_dir, f'{output_dir}/{split}_edits.json'), args.edits_granularity)

        # data without pnx edits
        print('Taking out the pnx from the edits...', flush=True)
        no_pnx_edits_data, pnx_edits_data = separate_pnx_edits(data)

        # compress no pnx edits
        print('Compressing no pnx edits...', flush=True)

        if split != 'train':
            compressed_no_pnx_data = compress_edits(test_data=no_pnx_edits_data, verify=False,
                                                    edits_granularity=args.edits_granularity,
                                                    compress_map_output_path=os.path.join(args.output_data_dir, f'{output_dir}/compress_map_nopnx.json'))
        else:
            compressed_no_pnx_data = compress_edits(train_data=no_pnx_edits_data, edits_granularity=args.edits_granularity,
                                                    verify=False,
                                                    compress_map_output_path=os.path.join(args.output_data_dir, f'{output_dir}/compress_map_nopnx.json'))
            

        write_json(path=os.path.join(args.output_data_dir, f'{output_dir}/{split}_edits_nopnx.json'),
                   data=compressed_no_pnx_data, edits_granularity=args.edits_granularity)
        write_tsv(path=os.path.join(args.output_data_dir, f'{output_dir}/{split}_edits_nopnx'),
                  data=compressed_no_pnx_data, edits_granularity=args.edits_granularity)
        get_stats(compressed_no_pnx_data, os.path.join(args.output_data_dir, f'{output_dir}/{split}_nopnx'),
                  edits_granularity=args.edits_granularity)

        # create pnx edits
        if args.create_pnx_edits:
            print('Creating pnx edits...', flush=True)
            pnx_edits_data = create_dataset_edits_parallel(compressed_no_pnx_data, tokenizer, num_workers=50)

        # compress pnx edits
        print('Compressing pnx edits...', flush=True)
        if split != 'train':
            compressed_pnx_data = compress_edits(test_data=pnx_edits_data, verify=True,
                                                 edits_granularity=args.edits_granularity,
                                                 compress_map_output_path=os.path.join(args.output_data_dir, f'{output_dir}/compress_map_pnx.json'))
        else:
            compressed_pnx_data = compress_edits(train_data=pnx_edits_data, edits_granularity=args.edits_granularity,
                                                 verify=True,
                                                 compress_map_output_path=os.path.join(args.output_data_dir, f'{output_dir}/compress_map_pnx.json'))
        
        # the compression 
        # compressed_pnx_data = compress_edits(data=pnx_edits_data, verify=True if args.create_pnx_edits else False,
                                            #  edits_granularity=args.edits_granularity)

        write_json(path=os.path.join(args.output_data_dir, f'{output_dir}/{split}_edits_pnx.json'),
                   data=compressed_pnx_data, edits_granularity=args.edits_granularity)
        write_tsv(path=os.path.join(args.output_data_dir, f'{output_dir}/{split}_edits_pnx'),
                  data=compressed_pnx_data, edits_granularity=args.edits_granularity)
        get_stats(compressed_pnx_data, os.path.join(args.output_data_dir, f'{output_dir}/{split}_pnx'),
                  edits_granularity=args.edits_granularity)



    if args.prune:
        prune_output_dir = args.pruned_output_dir
        nopnx_data = load_data(os.path.join(args.output_data_dir, f'{output_dir}/{split}_edits_nopnx.json'),
                               args.edits_granularity)
        pnx_data = load_data(os.path.join(args.output_data_dir, f'{output_dir}/{split}_edits_pnx.json'),
                             args.edits_granularity)

        if args.prune_cor:
            nopnx_pruned_data = prune_edits_corr(nopnx_data, k=args.k)
            pnx_pruned_data = prune_edits_corr(pnx_data, k=args.k)
        else:
            nopnx_pruned_data = prune_edits(nopnx_data, edits_granularity=args.edits_granularity, k=args.k)
            pnx_pruned_data = prune_edits(pnx_data, edits_granularity=args.edits_granularity, k=args.k)


        write_json(path=os.path.join(prune_output_dir, f'{output_dir}/{split}_edits_nopnx.json'),
                   data=nopnx_pruned_data, edits_granularity=args.edits_granularity)
        write_tsv(path=os.path.join(prune_output_dir, f'{output_dir}/{split}_edits_nopnx'),
                  data=nopnx_pruned_data, edits_granularity=args.edits_granularity)
        get_stats(nopnx_pruned_data, os.path.join(prune_output_dir, f'{output_dir}/{split}_nopnx'),
                  edits_granularity=args.edits_granularity)


        write_json(path=os.path.join(prune_output_dir, f'{output_dir}/{split}_edits_pnx.json'),
                   data=pnx_pruned_data, edits_granularity=args.edits_granularity)
        write_tsv(path=os.path.join(prune_output_dir, f'{output_dir}/{split}_edits_pnx'),
                  data=pnx_pruned_data, edits_granularity=args.edits_granularity)
        get_stats(pnx_pruned_data, os.path.join(prune_output_dir, f'{output_dir}/{split}_pnx'),
                  edits_granularity=args.edits_granularity)


