import re
from edit import SubwordEdit, Edit
from collections import defaultdict, Counter
import json
import copy
from string import punctuation
from camel_tools.utils.charsets import UNICODE_PUNCT_SYMBOL_CHARSET


def get_edits(data):
    """
    Gets all of the edits in data and their frequencies
    """
    edits = []
    for example in data:
        for edit in example['subword-edits-append']:
            edits.append(edit.edit)

    return dict(Counter(edits))


def compress_edits(data, verify=True):
    edits_freqs = get_edits(data)
    compressed_edits = []
    compressed_edits_map = dict()

    # get the compressed edits and their frequency over edit types
    for edit in edits_freqs:
        compressed_edit = compress_edit(edit)
        compressed_edits.extend(compressed_edit)
        compressed_edits_map[edit] = {e: edits_freqs[edit] for e in compressed_edit}


    compressed_edits_freqs = Counter(compressed_edits)

    # choose the compression for each edit based on a freq score
    final_compressed_edits_map = dict()

    for edit in compressed_edits_map:
        compressed_edits = compressed_edits_map[edit]
        # most_comm = max(compressed_edits, key=lambda x: x[1])[0]
        compressed_edit_freq = [(comp_edit, compressed_edits_freqs[comp_edit])
                                for comp_edit in compressed_edits]

        most_comm = max(compressed_edit_freq, key=lambda x: x[1])[0]
        final_compressed_edits_map[edit] = most_comm
    

    print(f'Uncompressed Edits', flush=True)
    print(len(edits_freqs))
    final_compressed_edits_freqs = Counter(final_compressed_edits_map.values())
    print(f'Compressed edits', flush=True)
    print(len(final_compressed_edits_freqs))


    # compressing the edits over the entire dataset
    compressed_data = []
    for example in data:
        example_compressed_edits = []
        example_edits = example['subword-edits-append']
        for subword_edit in example_edits:
            edit = SubwordEdit(subword_edit.subword,
                               final_compressed_edits_map[subword_edit.edit])
            example_compressed_edits.append(edit)
        
        # verifying the compression
        if verify:
            tokenized_src = [ex.subword for ex in example_edits]

            rewritten_src = apply_edits(tokenized_src, example_compressed_edits)
            if ' '.join(rewritten_src) != example['tgt']:
                import pdb; pdb.set_trace()

        _example = copy.deepcopy(example)
        _example['subword-edits-append'] = example_compressed_edits
        compressed_data.append(_example)
    
    return compressed_data



def compress_edit(edit):
    """
    Generates all possible compressions of an edit string by compressing sequences of Ks and Ds.
    """
    grouped_edits = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|A_\[.*?\]+|D+|K+|.', edit)
    
    candidates = [
        i for i, grouped_edit in enumerate(grouped_edits)
        if len(set(grouped_edit)) == 1 and grouped_edit[0] in ['K', 'D']
    ]

    compressed_candidates = [
        ''.join(grouped_edits[:candidate] + [f'{grouped_edits[candidate][0]}*'] + grouped_edits[candidate + 1:])
        for candidate in candidates
    ]

    return compressed_candidates if compressed_candidates else [edit]



def insert_to_append(subword_edits):
    """
    Converts insert edits to append edits when possible and compresses them.
    
    Args:
        subword_edits (list of SubwordEdit): List of subword edits.
    
    Returns:
        list of SubwordEdit: Updated subword edits with insertions transformed into appends.
    """
    processed_edits = []
    start_inserts = []

    subwords = [edit.subword for edit in subword_edits] # getting the subwords for book keeping

    for subword_edit in subword_edits:
        # Extract individual edits from subword edit
        subword_edit_parts = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|K\*|.', subword_edit.edit)
        all_inserts = all(edit.startswith('I_[') for edit in subword_edit_parts)

        if all_inserts:  # If entire subword edit consists of insertions, convert to append
            assert subword_edit.subword == ''

            append_edit = ''.join(re.sub(r'^I', r'A', edit) for edit in subword_edit_parts)

            if processed_edits:
                # Append to the last processed edit
                processed_edits[-1] = compress_appends(processed_edits[-1] + append_edit)
            else:
                # Store insertion edits that appear at the beginning
                start_inserts.append(append_edit)
        else:
            if start_inserts:
                # Add append edits before current edit
                processed_edits.append(compress_appends(''.join(start_inserts) + ''.join(subword_edit_parts)))
                start_inserts = []
            else:
                processed_edits.append(''.join(subword_edit_parts))

    # coverting the edits to objects
    subwords = [subword for subword in subwords if subword != '']

    assert len(processed_edits) == len(subwords)

    # Special case for appends at the beginning of the sequence
    if processed_edits[0].startswith('A') and re.sub(r'A_\[.*?\]', '', processed_edits[0]) == 'K':
        processed_edits[0] = processed_edits[0].replace('K', 'K' * len(subwords[0].replace('##', '')))

    processed_edits = [SubwordEdit(subword, edit) for subword, edit in zip(subwords, processed_edits)]

    return processed_edits



def compress_appends(subword_edit):
    """
    Compresses multiple consecutive append edits into a single edit.

    Args:
        subword_edit (SubwordEdit): Subword edit string to compress.

    Returns:
        str: Compressed subword edit string.
    """
    edits = re.findall(r'I_\[.*?\]+|A_\[.*?\]+|R_\[.*?\]+|K\*|.', subword_edit)
    compressed_edits = []
    append_buffer = []

    for edit in edits:
        if edit.startswith('A_'):
            # Collect append edits into a buffer
            append_buffer.append(re.sub(r'A_\[(.*?)\]', r'\1', edit))
        else:
            if append_buffer:
                # Compress and add the collected append edits
                compressed_edits.append(f"A_[{' '.join(append_buffer)}]")
                append_buffer = []
            compressed_edits.append(edit)

    if append_buffer:
        # Add remaining append edits
        compressed_edits.append(f"A_[{' '.join(append_buffer)}]")

    return ''.join(compressed_edits)



def apply_edits(tokenized_text, edits):
    assert len(tokenized_text) == len(edits)

    rewritten_txt = []

    for subword, edit in zip(tokenized_text, edits):
        try:
            rewritten_subword = edit.apply(subword)
        except:
            import pdb; pdb.set_trace()

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


def apply_edits_subwords(tokenized_text, edits, pruned_edits):
    assert len(tokenized_text) == len(edits) == len(pruned_edits)

    rewritten_txt = []
    applied_edits = []
    _pruned_edits = []

    for subword, edit, pruned_edit in zip(tokenized_text, edits, pruned_edits):
        try:
            rewritten_subword = edit.apply(subword)
            if rewritten_subword.replace('##', '') != '':
                applied_edits.append(edit)
                _pruned_edits.append(pruned_edit)
        except:
            import pdb; pdb.set_trace()

        edit_ops = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|A_\[.*?\]+|D+|K+|.', edit.edit)

        if 'M' in edit_ops: # merge
            rewritten_txt[-1] = rewritten_txt[-1] + rewritten_subword
            if rewritten_subword != '': # MD* case
                applied_edits.pop()
                _pruned_edits.pop()
        else:
            if rewritten_subword.replace('##', '') != '':
                rewritten_txt.append(rewritten_subword)

    try:
        assert len(rewritten_txt) == len(applied_edits) == len(_pruned_edits)
    except:
        import pdb; pdb.set_trace()

    return rewritten_txt, _pruned_edits


def prune_edits(data, k):
    all_edits = []
    for example in data:
        example_edits = example['subword-edits-append']
        for subword_edit in example_edits:
            subword, edit = subword_edit.subword, subword_edit.edit
            all_edits.append(edit)
    
    edits_cnt = Counter(all_edits)
    _pruned_data = []

    for example in data:
        _example = copy.deepcopy(example)
        example_edits = _example['subword-edits-append']
        pruned_edits = []

        for subword_edit in example_edits:
            subword, edit = subword_edit.subword, subword_edit.edit
            if edits_cnt[edit] > k:
                pruned_edits.append(subword_edit)
            else:
                pruned_edits.append(SubwordEdit(subword, 'K*'))
        
        _example['subword-edits-append'] = pruned_edits
        _pruned_data.append(_example)
    
    return _pruned_data



def prune_edits_corr(data, k):
    all_edits = []
    for example in data:
        example_edits = example['subword-edits-append']
        for subword_edit in example_edits:
            subword, edit = subword_edit.subword, subword_edit.edit
            all_edits.append(edit)
    
    edits_cnt = Counter(all_edits)
    _pruned_data = []

    # we will prune the edits and apply the edits that got pruned

    for example in data:
        _example = copy.deepcopy(example)
        example_edits = _example['subword-edits-append']
        pruned_edits = []
        edits_to_apply = []
        pruned = False
        for subword_edit in example_edits:
            subword, edit = subword_edit.subword, subword_edit.edit
            if edits_cnt[edit] > k:
                pruned_edits.append(subword_edit)
                edits_to_apply.append(SubwordEdit(subword, 'K*'))
            else:
                pruned_edits.append(SubwordEdit(subword, 'K*'))
                edits_to_apply.append(subword_edit)
                pruned = True
        
        if pruned:
            tokenized_src = [ex.subword for ex in example_edits]
            # we will apply the edits that got removed and replace them by K*
            # this has to be done carefully so that the new subwords match the number of pruned edits
            new_tokenized_src, pruned_edits = apply_edits_subwords(tokenized_src, edits_to_apply, pruned_edits)
            _example['subword-edits-append'] = [SubwordEdit(subword, edit.edit) for
                                                subword, edit in zip(new_tokenized_src, pruned_edits)]
        else:
            _example['subword-edits-append'] = pruned_edits

        _pruned_data.append(_example)
    
    return _pruned_data
    


def write_json(path, data):
    with open(path, mode='w') as f:
        for example in data:
            src = example['src']
            tgt = example['tgt']
            word_level_align = example['word-level-align']
            char_level_align = example['char-level-align']
            word_edits = [edit.to_json_str() for edit in example['word-edits']]
            subword_edits = [edit.to_json_str() for edit in example['subword-edits']]
            subword_edits_append =  [edit.to_json_str() for edit in example['subword-edits-append']]


            f.write(json.dumps({'src': src, 'tgt': tgt,
                                'word-level-align': word_level_align, 'char-level-align': char_level_align,
                                'word-edits': word_edits, 'subword-edits': subword_edits,
                                'subword-edits-append': subword_edits_append}, ensure_ascii=False))
            f.write('\n')


def write_tsv(path, data):
    with open(f'{path}.tsv', mode='w') as f1, open(f'{path}.append.tsv', mode='w') as f2:

        for example in data:
            subword_edits = example['subword-edits']
            subword_edits_append = example['subword-edits-append']

            for subword_edit in subword_edits:
                f1.write(f'<s>{subword_edit.subword}<s>\t<s>{subword_edit.edit}<s>')
                f1.write('\n')

            for subword_edit in subword_edits_append:
                f2.write(f'<s>{subword_edit.subword}<s>\t<s>{subword_edit.edit}<s>')
                f2.write('\n')

            f1.write('\n')
            f2.write('\n')


def write_tsv_word_edits(path, data):
    with open(f'{path}.tsv', mode='w') as f1, open(f'{path}.append.tsv', mode='w') as f2:

        for example in data:
            word_edits = example['word-edits']

            for edit in word_edits:
                f1.write(f'<s>{edit.aligned_word}<s>\t<s>{edit.edit}<s>')
                f1.write('\n')

            f1.write('\n')
            f2.write('\n')


def load_data(path):
    data = []
    with open(path) as f:
        for line in f.readlines():
            example = json.loads(line)
            word_edits = [Edit.from_json(json.loads(edit)) for edit in example['word-edits']]
            subword_edits = [SubwordEdit.from_json(json.loads(edit)) for edit in example['subword-edits']]
            subword_edits_append = [SubwordEdit.from_json(json.loads(edit)) for edit in example['subword-edits-append']]
        
            data.append({'src': example['src'], 'tgt': example['tgt'],
                         'word-level-align': example['word-level-align'],
                         'char-level-align': example['char-level-align'],
                         'word-edits': word_edits, 'subword-edits': subword_edits,
                         'subword-edits-append': subword_edits_append})
    return data


def get_stats(data, path):
    subword_edits = defaultdict(list)
    subword_edits_append = defaultdict(list)

    for example in data:
        ex_subword_edits = example['subword-edits']
        ex_subword_edits_append = example['subword-edits-append']

        for edit in ex_subword_edits:
            subword_edits[edit.edit].append(edit)
        
        for edit in ex_subword_edits_append:
            subword_edits_append[edit.edit].append(edit)

    get_subwords_len(data)


    with open(f'{path}_stats.tsv', mode='w') as f1, open(f'{path}_stats.append.tsv', mode='w') as f2:
        f1.write('Edit\t#Edits\tFreq\n')
        for edit in subword_edits:
            edits = re.findall(r'I_\[.*?\]+|A_\[.*?\]+|R_\[.*?\]+|K\*|D\*|.', edit)
            f1.write(f'<s>{edit}<s>\t<s>{len(edits)}<s>\t<s>{len(subword_edits[edit])}<s>\n')

        f2.write('Edit\t#Edits\tFreq\n')
        for edit in subword_edits_append:
            edits = re.findall(r'I_\[.*?\]+|A_\[.*?\]+|R_\[.*?\]+|K\*|.', edit)
            f2.write(f'<s>{edit}<s>\t<s>{len(edits)}<s>\t<s>{len(subword_edits_append[edit])}<s>\n')



def get_subwords_len(data):
    subwords_len = []

    for example in data:
        ex_subword_edits = example['subword-edits-append']
        for edit in ex_subword_edits:
            subword = edit.subword.replace('##', '')
    
            if subword:
                subwords_len.append(len(subword))
    
    len_cnts = Counter(subwords_len)
    sorted_cnts = sorted(len_cnts.items(), key=lambda x: x[1], reverse=True)
    sorted_cnts = {x[0]: x[1] for x in sorted_cnts}

    for _len, freq in sorted_cnts.items():
        print(f'{_len}\t{freq}')
    print()



PNX = punctuation + ''.join(list(UNICODE_PUNCT_SYMBOL_CHARSET)) + '&amp;'
pnx_patt = re.compile(r'(['+re.escape(PNX)+'])')



def separate_pnx_edits(data):
    no_pnx_edits_data = []
    pnx_error_subwords = []

    for example in data:
        example_no_pnx_edits = []
        example_edits = example['subword-edits-append']
        for i, subword_edit in enumerate(example_edits):
            sep_pnx = separate_pnx_edit(subword_edit.edit)
            pnx_edit, no_pnx_edit = sep_pnx['pnx_edit'], sep_pnx['no_pnx_edit']

            edit = SubwordEdit(subword_edit.subword, no_pnx_edit)
            example_no_pnx_edits.append(edit)

   
        tokenized_src = [ex.subword for ex in example_edits]

        rewritten_src = apply_edits(tokenized_src, example_no_pnx_edits)
        # no_pnx_tgt = pnx_patt.sub('', example['tgt'])
        # no_pnx_tgt = re.sub(' +', ' ', no_pnx_tgt).strip()

        # no_pnx_rewritten_src = pnx_patt.sub('', ' '.join(rewritten_src))
        # no_pnx_rewritten_src = re.sub(' +', ' ', no_pnx_rewritten_src).strip()

        # if no_pnx_rewritten_src != no_pnx_tgt:
        #     import pdb; pdb.set_trace()

        _example = copy.deepcopy(example)
        _example['cor-no-pnx'] = ' '.join(rewritten_src)
        _example['subword-edits-append'] = example_no_pnx_edits
        no_pnx_edits_data.append(_example)
    
    return no_pnx_edits_data


def separate_pnx_edit(edit):
    """
    Given an edit, returns two edits. One for pnx edits and one for no pnx edits.
    """
    grouped_edits = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|A_\[.*?\]+|D+|K+|.', edit)

    pnx_edit = ''
    no_pnx_edit = '' 
    found_pnx = False

    for g_edit in grouped_edits:
        if g_edit.startswith('A_[') or g_edit.startswith('I_[') or g_edit.startswith('R_['):
            op = g_edit[0]
            seq = re.sub(op + r'_\[(.*?)\]', r'\1', g_edit)
            seq = re.sub(' +', '', seq)
            if pnx_patt.findall(seq) and ''.join(pnx_patt.findall(seq)) == seq:
                pnx_edit += g_edit
                found_pnx = True 
                if op == 'R':
                    no_pnx_edit += 'K'
            else:
                no_pnx_edit += g_edit
                if g_edit.startswith('R_['):
                    pnx_edit += 'K'

        elif g_edit:
            no_pnx_edit += g_edit
            if not (g_edit.startswith('I') and g_edit.startswith('M') and g_edit.startswith('A')):
                pnx_edit += 'K' * len(g_edit)
    
    if found_pnx == False:
        pnx_edit = ''


    re_edit = reconstruct_edit(pnx_edit=pnx_edit, no_pnx_edit=no_pnx_edit)
    assert  re_edit == edit
    return {'no_pnx_edit': no_pnx_edit, 'pnx_edit': pnx_edit}


def reconstruct_edit(pnx_edit, no_pnx_edit):
    def parse_edits(edit_string):
        """Parse edits into grouped operations."""
        return re.findall(r'I_\[.*?\]+|R_\[.*?\]+|A_\[.*?\]+|D|K|.', edit_string)

    def is_insert_or_append(edit):
        """Check if the edit is an insert or append operation."""
        return edit.startswith('I') or edit.startswith('A')

    def is_replace(edit):
        """Check if the edit is a replace operation."""
        return edit.startswith('R')

    # Parse the edits and initialize counters
    pnx_grouped_edits = parse_edits(pnx_edit)
    no_pnx_grouped_edits = parse_edits(no_pnx_edit)
    pnx_edit_cnts = Counter(pnx_grouped_edits)
    no_pnx_edit_cnts = Counter(edit for edit in no_pnx_grouped_edits if not is_insert_or_append(edit))

    
    i, j = 0, 0
    reconstructed_edit = ""

    # Merge edits
    while i < len(pnx_grouped_edits) and j < len(no_pnx_grouped_edits):
        pnx_edit = pnx_grouped_edits[i]
        no_pnx_edit = no_pnx_grouped_edits[j]

        # adding no pnx edit if pnx_edit is K and the no_pnx_edit is in [K, D, M, R]
        if pnx_edit == 'K' and (no_pnx_edit in ['K', 'D', 'M'] or is_replace(no_pnx_edit)):
            reconstructed_edit += no_pnx_edit
            pnx_edit_cnts[pnx_edit] -= 1
            no_pnx_edit_cnts[no_pnx_edit] -= 1
            i += 1
            j += 1

        # adding pnx edit if pnx edit is replace and no pnx edit is K
        elif is_replace(pnx_edit) and no_pnx_edit == 'K':
            reconstructed_edit += pnx_edit
            pnx_edit_cnts[pnx_edit] -= 1
            no_pnx_edit_cnts[no_pnx_edit] -= 1
            i += 1
            j += 1

        elif is_insert_or_append(pnx_edit):
            if pnx_edit_cnts['K'] != 0 and sum(no_pnx_edit_cnts.values()) == pnx_edit_cnts['K']:
                reconstructed_edit += pnx_edit
                i += 1
            else:
                reconstructed_edit += no_pnx_edit
                j += 1
        else:
            reconstructed_edit += no_pnx_edit
            j += 1


    # adding remaining edits
    reconstructed_edit += ''.join(no_pnx_grouped_edits[j:])
    reconstructed_edit += ''.join(pnx_grouped_edits[i:])
    
    return reconstructed_edit
