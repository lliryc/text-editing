import argparse
from edits.edit import SubwordEdit

def read_tags_subwords(path):
    tokens = []
    all_tokens = []
    tags = []
    all_tags = []

    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                all_tokens.append(tokens)
                all_tags.append(tags)
                tokens = []
                tags = []

            else:
                line = line.split('\t')
                tokens.append(line[0])
                tags.append(line[1])

    if tokens:
        all_tokens.append(tokens)

    if tags:
        all_tags.append(tags)

    return all_tokens, all_tags


def read_tags(path):
    tags = []
    all_tags = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                all_tags.append(tags)
                tags = []
            else:
                tags.append(line)
 
    if tags:
        all_tags.append(tags)


    return all_tags


def read_subwords(path):
    tokens = []
    all_tokens = []
    with open(path) as f:
        for line in f.readlines():
            line = line.strip()
            if not line:
                all_tokens.append(tokens)
                tokens = []
            else:
                line = line.split('\t')
                tokens.append(line[0])

    if tokens:
        all_tokens.append(tokens)


    return all_tokens


def apply_edits(subwords, edits):
    assert len(subwords) == len(edits)

    rewritten_sents = []
    non_app_edits = []

    for i, (sent_subwords, sent_edits) in enumerate(zip(subwords, edits)):
        assert len(sent_subwords) == len(sent_edits)
        rewritten_sent = []

        for subword, edit in zip(sent_subwords, sent_edits):
            edit = SubwordEdit(subword, edit)

            if edit.is_applicable(subword):
                rewritten_subword = edit.apply(subword)
                rewritten_sent.append(rewritten_subword)
            else:
                non_app_edits.append({'subword': subword, 'edit': edit.to_json_str()})
                rewritten_sent.append(subword)

        rewritten_sents.append(rewritten_sent)


    return rewritten_sents, non_app_edits

def write_data(path, data):
    with open(path, mode='w') as f:
        for example in data:
            for subword in example:
                f.write(f'<s>{subword}<s>')
                f.write('\n')
            f.write('\n')

subwords, edits = read_tags_subwords('/scratch/ba63/arabic-text-editing/gec_data/zaebuc/pnx_sep/nopnx/zaebuc/dev_nopnx.txt')

# subwords, edits = read_tags_subwords('/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep/nopnx/qalb14/dev_nopnx.txt')
# edits = read_tags('/scratch/ba63/arabic-text-editing/tagger_data/qalb14/dev.txt')
# subwords = read_subwords('/scratch/ba63/arabic-text-editing/tagger_data/qalb14/dev.txt')

assert len(edits) == len(subwords)

rewritten_sents, non_app_edits = apply_edits(subwords, edits)
# write_data('/scratch/ba63/arabic-text-editing/tagger_data/pnx_sep/nopnx/qalb14/dev_nopnx.tgt.txt', rewritten_sents)
write_data('/scratch/ba63/arabic-text-editing/gec_data/zaebuc/pnx_sep/nopnx/zaebuc/dev_nopnx.tgt.txt', rewritten_sents)

# import pdb; pdb.set_trace()
