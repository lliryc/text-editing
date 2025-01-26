from data_utils import Dataset, read_txt_file, write_data
from collections import defaultdict
import dill as pickle
import re


def build_ngrams(sentence, pad_right=False, pad_left=False, ngrams=1):
    """
    Args:
     - sentence (list of str): a list of words.
     - ngrams (int): 2 for bigrams, 3 for trigrams, etc.
     - pad_right (bool): adding </s> to the end of sentence
     - pad_left (bool): adding <s> to the beginning of sentence
    Returns:
     - ngrams of the sentence (list of tuples)
    """

    if pad_right:
        sentence = sentence + ['</s>'] * (ngrams - 1)
    if pad_left:
        sentence = ['<s>'] * (ngrams - 1) + sentence
    return [tuple(sentence[i - (ngrams - 1): i + 1])
            for i in range(ngrams - 1, len(sentence))]


class WordLookup():
    def __init__(self, model, ngrams, backoff=True):
        self.model = model
        self.ngrams = ngrams
        self.backoff = backoff

    @classmethod
    def build_model(cls, dataset, ngrams=1, backoff=True, min_freq=10):
        """
        Args:
            - dataset (Dataset obj)
            - backoff (bool): backoff to a lower order ngram during lookup.
            - ngrams (int): number of ngrams
        Returns:
            - cbr model (default dict): The cbr model where the
            keys are source_word and vals
            are target_word
        """

        model = defaultdict(lambda: defaultdict(lambda: 0))
        context = dict()

        for ex in dataset.examples:
            src_tokens = ex.src_tokens
            tgt_tokens = ex.tgt_tokens


            # getting counts of all ngrams
            # until ngrams == 1
            for i in range(ngrams):
                src_tokens_ngrams = build_ngrams(src_tokens, ngrams=i + 1,
                                                 pad_left=True)

                assert len(src_tokens) == len(src_tokens_ngrams)

                for j, tgt_w in enumerate(tgt_tokens):
                    src_ngram = src_tokens_ngrams[j]
                    # counts of (t_w, s_w)
                    model[src_ngram][tgt_w] += 1
                    # counts of (s_w)
                    context[src_ngram] = 1 + context.get(src_ngram, 0)


        # turning the counts into probs
        for sw in list(model.keys()): # Use list() to safely modify while iterating
            for tgt_w in list(model[sw].keys()):
                if model[sw][tgt_w] < min_freq:
                    del model[sw][tgt_w]  # Remove low-frequency target words
                else:
                    model[sw][tgt_w] /= float(context[sw])

                if not model[sw]:  # Remove source words with no valid target words
                    del model[sw]

        return cls(model, ngrams, backoff)


    def __getitem__(self, sw_gt):
        context = sw_gt

        if self.backoff:
            # keep backing-off until a context is found
            for i in range(self.ngrams):
                if context[i:] in self.model:
                    return dict(self.model[context[i:]])
        else:
            if context in self.model:
                return dict(self.model[context])

        # worst case, return None
        return None


def rewrite_sent(sentence, model):
    rewritten_sent = []
    words = sentence.split()

    tokens_ngrams = build_ngrams(words,
                                 ngrams=model.ngrams,
                                 pad_left=True)
    assert len(words) == len(tokens_ngrams)

    for i, ngram in enumerate(tokens_ngrams):
        candidates = model[ngram]
        if candidates:
            rewritten_word = max(candidates.items(), key=lambda x: x[1])[0]
            rewritten_sent.append(rewritten_word)
        else:
            rewritten_sent.append(words[i])
    
    rewritten_sent = ' '.join(rewritten_sent)
    rewritten_sent = re.sub(' +', ' ', rewritten_sent)
    return rewritten_sent


def rewrite(data, model):
    rewritten_data = []
    for sent in data:
        rewritten_sent = rewrite_sent(sent, model)
        rewritten_data.append(rewritten_sent)
    
    return rewritten_data


data = Dataset('/home/ba63/gec-release/data/alignments/modeling/qalb14/qalb14_train.nopnx.txt')
# data = Dataset('mix_train.nopnx.txt')
test_data = read_txt_file('/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/dev/QALB-2014-L1-Dev.sent.no_ids.clean.dediac')
# test_data = read_txt_file('/home/ba63/gec-release/data/gec/ZAEBUC-v1.0/data/ar/dev/dev.sent.raw.pnx.tok.dediac')
# test_data = read_txt_file('/home/ba63/gec-release/data/gec/QALB-0.9.1-Dec03-2021-SharedTasks/data/2014/train/QALB-2014-L1-Train.sent.no_ids.clean.dediac')



model = WordLookup.build_model(data, ngrams=2, backoff=True, min_freq=0)

rewritten_data = rewrite(test_data, model)
write_data('qalb14_dev.mle.txt', rewritten_data)