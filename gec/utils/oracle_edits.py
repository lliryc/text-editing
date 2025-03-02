from edits.edit import SubwordEdit
from collections import Counter
import re

from gec.utils.postprocess import remove_pnx, pnx_tokenize, space_clean


def read_data(path):
    example_edits = []
    all_edits = []

    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if line:
                subword, label = line.split('\t')
                subword = subword.replace('<s>', '')
                label = label.replace('<s>', '')
                subword_edit = SubwordEdit(subword=subword, raw_subword=subword, edit=label)
                example_edits.append(subword_edit)
            else:
                all_edits.append(example_edits)
                example_edits = []
        
        if example_edits:
            all_edits.append(example_edits)
    
    return all_edits


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

    detok_rewritten_sents = [detokenize_sent(sent) for sent in rewritten_sents_merge]
    return detok_rewritten_sents


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


def detokenize_sent(sent):
    detokenize_sent = []
    for subword in sent:
        if subword.startswith('##'):
            detokenize_sent[-1] = detokenize_sent[-1] + subword.replace('##', '')
        else:
            detokenize_sent.append(subword)

    return ' '.join(detokenize_sent)


def lookup_edits(train_edits, test_edits, comp=False, pnx_prepoc=False, clean_space=False, delete_pnx=False):
    all_train_edits = [edit.edit for ex in train_edits for edit in ex]
    all_train_edits = Counter(all_train_edits)
    print(f'Train Edits:       {len(all_train_edits)}')
    oov_edit = 0
    edits_to_apply = []
    for example in test_edits:
        ex_apply_edits = []
        for edit in example:
            if edit.edit in all_train_edits:
                ex_apply_edits.append(SubwordEdit(subword=edit.subword, raw_subword=edit.raw_subword, edit=edit.edit))
            else:
                ex_apply_edits.append(SubwordEdit(subword=edit.subword, raw_subword=edit.raw_subword, edit='K*' if comp else 'K'))
                oov_edit += 1 
            
        edits_to_apply.append(ex_apply_edits)
    
    print(f'OOV Edit (Token):  {oov_edit}')
    
    rewritten_dev = rewrite(edits_to_apply)

    if pnx_prepoc:
        rewritten_dev = pnx_tokenize(rewritten_dev)
    
    if clean_space:
        rewritten_dev = space_clean(rewritten_dev)

    if delete_pnx:
        rewritten_dev = remove_pnx(rewritten_dev)

    return rewritten_dev


def write_data(path, data):
    with open(path, mode='w') as f:
        for example in data:
            f.write(example)
            f.write('\n')


def process_data(dataset, edits_dir, prune_levels=None, pnx_seg=False, clean_space=False, pnx_preproc=False):
    """
    Processes a dataset for different granularity, compression, and pruning levels.
    """
    print(dataset)
    print('-------------------')
    # Process different granularities and compression settings
    for gran in ['word', 'subword']:
        for comp in ['compressed', 'no_compressed']:
            comp_flag = comp == 'compressed'
            print(f'Gran: {gran}\nComp.: {comp_flag}\nPrune: 0')
            
            train_path = f'{edits_dir}/edits_{comp}/{dataset}-arabertv02/{gran}-level/train_edits.modeling.tsv'
            dev_path = f'{edits_dir}/edits_{comp}/{dataset}-arabertv02/{gran}-level/dev_edits.modeling.tsv'
            
            train_data = read_data(train_path)
            dev_data = read_data(dev_path)
            
            rewritten_dev = lookup_edits(train_edits=train_data, test_edits=dev_data, comp=comp_flag,
                                         clean_space=clean_space, pnx_prepoc=pnx_preproc)
            write_data(f'predictions/oracle/{dataset}/{dataset}_dev_{gran}_{comp}.txt', rewritten_dev)
            print()
    
    # Process subword-level compressed pruning
    if prune_levels:
        dev_comp = read_data(f'{edits_dir}/edits_compressed/{dataset}-arabertv02/subword-level/dev_edits.modeling.tsv')
        for k in prune_levels:
            print(f'Gran: Subword\nComp.: True\nPrune: {k}')
            
            train_path = f'{edits_dir}/edits_compressed_prune_{k}/{dataset}-arabertv02/subword-level/train_edits.modeling.tsv'
            train_data = read_data(train_path)
            
            rewritten_dev = lookup_edits(train_edits=train_data, test_edits=dev_comp, comp=True,
                                         clean_space=clean_space, pnx_prepoc=pnx_preproc)
            write_data(f'predictions/oracle/{dataset}/prune/{dataset}_dev_subword_compressed_prune_{k}.txt', rewritten_dev)
            print()
    
    # Process pnx segmentation if required
    if pnx_seg:
        dev_nopnx_path = f'{edits_dir}/edits_compressed_pnx_sep/{dataset}-arabertv02/subword-level/dev_edits_nopnx_edits.modeling.tsv'
        dev_pnx_path = f'{edits_dir}/edits_compressed_pnx_sep/{dataset}-arabertv02/subword-level/dev_edits_pnx_edits.modeling.tsv'
        
        dev_nopnx = read_data(dev_nopnx_path)
        dev_pnx = read_data(dev_pnx_path)
        
        for k in [0, 10, 20, 30]:
            if k == 0:
                train_nopnx_path = f'{edits_dir}/edits_compressed_pnx_sep/{dataset}-arabertv02/subword-level/train_edits_nopnx_edits.modeling.tsv'
                train_pnx_path = f'{edits_dir}/edits_compressed_pnx_sep/{dataset}-arabertv02/subword-level/train_edits_pnx_edits.modeling.tsv'
            else:
                train_nopnx_path = f'{edits_dir}/edits_compressed_pnx_sep_prune_{k}/{dataset}-arabertv02/subword-level/train_edits_nopnx_edits.modeling.tsv'
                train_pnx_path = f'{edits_dir}/edits_compressed_pnx_sep_prune_{k}/{dataset}-arabertv02/subword-level/train_edits_pnx_edits.modeling.tsv'
            
            train_nopnx = read_data(train_nopnx_path)
            rewritten_nopnx = lookup_edits(train_edits=train_nopnx, test_edits=dev_nopnx, comp=True, pnx_prepoc=True, delete_pnx=True)
            
            train_pnx = read_data(train_pnx_path)
            rewritten_pnx = lookup_edits(train_edits=train_pnx, test_edits=dev_pnx, comp=True, pnx_prepoc=True)
            if k == 0:
                write_data(f'predictions/oracle/{dataset}/pnx_seg/{dataset}_dev_subword_compressed_nopnx.txt', rewritten_nopnx)
                write_data(f'predictions/oracle/{dataset}/pnx_seg/{dataset}_dev_subword_compressed_pnx.txt', rewritten_pnx)
            else:
                write_data(f'predictions/oracle/{dataset}/pnx_seg/{dataset}_dev_subword_compressed_prune_{k}_nopnx.txt', rewritten_nopnx)
                write_data(f'predictions/oracle/{dataset}/pnx_seg/{dataset}_dev_subword_compressed_prune_{k}_pnx.txt', rewritten_pnx)


if __name__ == '__main__':
    process_data('qalb14', '../data/msa-gec/edits/qalb14', prune_levels=[10, 20, 30], pnx_seg=True,
                 clean_space=False, pnx_preproc=True)
    process_data('zaebuc', '../data/msa-gec/edits/zaebuc', prune_levels=[10, 20, 30], pnx_seg=True,
                 clean_space=False, pnx_preproc=True)
    process_data('madar', '../data/da-gec/edits/madar', prune_levels=[10, 20, 30], clean_space=True,
                 pnx_preproc=False)