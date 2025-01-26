from collections import defaultdict, Counter
import pickle
import re
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from edits.edit import Edit, SubwordEdit
from util.postprocess import pnx_tokenize, remove_pnx
import json


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


def load_data(path):
    data = []
    with open(path) as f:
        for line in f.readlines():
            example = json.loads(line)
            word_edits = [Edit.from_json(json.loads(edit)) for edit in example['word-edits']]
            subword_edits = [SubwordEdit.from_json(json.loads(edit)) for edit in example['subword-edits']]
            subword_edits_append = [SubwordEdit.from_json(json.loads(edit)) for edit in example['subword-edits-append']]
        
            data.append({'src': example['src'], 'tgt': example['tgt'],
                         'word-edits': word_edits, 'subword-edits': subword_edits,
                         'subword-edits-append': subword_edits_append})
    return data


def read_data(file_path):
    examples = []

    with open(file_path, encoding="utf-8") as f:
        example = []
        for line in f:
            if line == "" or line == "\n":
                examples.append(example)
                example = []
            else:
                subword, edit = line.split("\t")
                example.append(SubwordEdit(subword, edit))

        if example:
            examples.append(example)

    return examples



class EditGivenSubword:
    """
    P(edit | error_type)
    """
    def __init__(self, model, ngrams, backoff=True):
        self.model = model
        self.ngrams = ngrams
        self.backoff = backoff


    @classmethod
    def build_model(cls, dataset, ngrams=1, backoff=True, ignore_insert=False):
        """
        Args:
            - dataset (Dataset obj)
            - backoff (bool): backoff to a lower order ngram during lookup.
            - ngrams (int): number of ngrams
        Returns:
            - cbr model (default dict): The cbr model where the
            key is the edit and value is the error type
        """

        model = defaultdict(lambda: defaultdict(lambda: 0))
        context = dict()

        for ex in dataset:
            subwords_edits = [subword_edit for subword_edit in ex]
            subwords = [e.subword for e in subwords_edits]
            edits = [e.edit for e in subwords_edits]
    
            if ignore_insert:
                edits = ignore_insertions(edits)
    
            assert len(edits) == len(subwords)

            # getting counts of all ngrams
            # until ngrams == 1
            for i in range(ngrams):

                subwords_ngrams = build_ngrams(subwords, ngrams=i + 1, pad_left=True)

                assert len(edits) == len(subwords_ngrams)

                for j, edit in enumerate(edits):
                    subword_ngram = subwords_ngrams[j]

                    if edit != 'K*':
                        # counts of (edit, error_type)
                        model[subword_ngram][edit] += 1
                        # counts of (edit)
                        context[subword_ngram] = 1 + context.get(subword_ngram, 0)

        # turning the counts into probs
        for subword in model:
            for edit in model[subword]:
                model[subword][edit] /= float(context[subword])

        return cls(model, ngrams, backoff)


    def rewrite(self, input_subwords):

        rewritten_sent = []
        rewritten_sent_merge = []
        pred_edits = []
    

        input_subwords_ngrams = build_ngrams(input_subwords, ngrams=self.ngrams,
                                             pad_left=True)

        assert len(input_subwords_ngrams) == len(input_subwords)
    
        for i, subwords_ngram in enumerate(input_subwords_ngrams):
            edits = self.model[subwords_ngram]
            input_subword = input_subwords[i]

            # confirming that all edits are applicable
            applicable_edits = self.filter(input_subword, edits)
            assert dict(edits) == applicable_edits

            if edits:
                selected_edit = max(edits.items(), key=lambda x: x[1])[0]
                if selected_edit != 'K*':
                    print(f'{input_subword}\t{selected_edit}')
                edit = SubwordEdit(input_subword, selected_edit)
                rewritten_subword = edit.apply(input_subword)
    
                pred_edits.append(selected_edit)
                rewritten_sent.append(rewritten_subword)
            
            else: # (OOV --> Keep)
                pred_edits.append('K*')
                rewritten_sent.append(input_subword)
        
        assert len(pred_edits) == len(rewritten_sent)

        rewritten_sent = self.resolve_merges(rewritten_sent, pred_edits)
        detok_rewritten_sent = detokenize_sent(rewritten_sent)

        return  rewritten_sent, pred_edits, detok_rewritten_sent


    def filter(self, subword, edits):
        applicable_edits = {}

        for edit, prop in edits.items():
            if SubwordEdit(subword, edit).is_applicable(subword):
                applicable_edits[edit] = prop
        
        return applicable_edits


    def resolve_merges(self, sent, edits):
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


    def __getitem__(self, subword_ngram):

        if self.backoff:
            # keep backing-off until a context is found
            for i in range(self.ngrams):
                if subword_ngram[i:] in self.model:
                    return dict(self.model[subword_ngram[i:]])
        else:
            if subword_ngram in self.model:
                return dict(self.model[subword_ngram])
        # worst case, return None
        return None

    def __len__(self):
        return len(self.model)

    @staticmethod
    def load_model(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
        


class ContextualTokenClassifier:
    def __init__(self, context_size=1):
        """
        Args:
            context_size (int): Number of subwords before and after the target subword.
        """
        self.context_size = context_size
        self.context_edit_probs = {}
        self.default_edit = "K*"  # Fallback edit for unseen contexts


    def _get_context(self, sequence, index, window_size):
        """
        Extract the context window of a given size for a subword at a given index.
        Args:
            sequence (list of str): The sequence of subwords.
            index (int): Index of the target subword.
            window_size (int): Size of the context window.
        Returns:
            tuple of str: Context window.
        """
        start = max(0, index - window_size)
        end = min(len(sequence), index + window_size + 1)
        return tuple(sequence[start:end])


    def train(self, training_data):
        """
        Train the lookup classifier using context-edit pairs from the training data.
        Args:
            training_data (list of tuples): List of (sequence, edits) pairs.
        """
        # Count edits for each context
        context_counts = defaultdict(Counter)
        for ex in training_data:
            subwords_edits = [subword_edit for subword_edit in ex]
            subwords = [e.subword for e in subwords_edits]
            edits = [e.edit for e in subwords_edits]
            edits = ignore_insertions(edits)
    
            assert len(edits) == len(subwords)
            for i, (subword, edit) in enumerate(zip(subwords, edits)):
                if edit != 'K*':
                    for window_size in range(self.context_size, -1, -1):
                        context = self._get_context(subwords, i, window_size)
                        context_counts[context][edit] += 1

        # Compute probabilities
        self.context_edit_probs = {
            context: {edit: count / sum(edit_counts.values())
                      for edit, count in edit_counts.items()}
            for context, edit_counts in context_counts.items()
        }

        # self.context_edit_probs = defaultdict(Counter)
    
        # for context, edit_cnts in context_counts.items():
        #     for edit, cnt in edit_cnts.items():
        #         if cnt < 100:
        #             continue
        #         self.context_edit_probs[context][edit] = cnt / sum(edit_cnts.values())
        
        # import pdb; pdb.set_trace()

    def predict(self, sequence):
        """
        Predict the edit sequence for a given sequence of subwords.
        Args:
            sequence (list of str): List of input subwords.
        Returns:
            list of str: Predicted edits for each subword.
        """
        predictions = []
        for i, subword in enumerate(sequence):
            predicted_edit = self.default_edit
            for window_size in range(self.context_size, -1, -1):
                context = self._get_context(sequence, i, window_size)
                if context in self.context_edit_probs:
                    predicted_edit = max(self.context_edit_probs[context], key=self.context_edit_probs[context].get)
                    break
            predictions.append(predicted_edit)
        return predictions



def rewrite(sent_subwords, sent_edits):
    assert len(sent_subwords) == len(sent_edits)
    rewritten_sent = []

    for subword, edit in zip(sent_subwords, sent_edits):
        edit = SubwordEdit(subword, edit)

        if edit.is_applicable(subword):
            rewritten_subword = edit.apply(subword)
            rewritten_sent.append(rewritten_subword)
        else:
            # non_app_edits.append({'subword': subword, 'edit': edit.to_json_str()})
            rewritten_sent.append(subword)

    rewritten_sent_merge = resolve_merges(rewritten_sent, sent_edits)
    detok_rewritten_sent = detokenize_sent(rewritten_sent_merge)
    return detok_rewritten_sent, rewritten_sent_merge


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



def ignore_insertions(edits):
    _edits = []

    for edit in (edits):
        ops = re.findall(r'I_\[.*?\]+|A_\[.*?\]+|R_\[.*?\]+|K\*|D\*|.', edit)
        _edits.append(''.join([op for op in ops if not op.startswith('A_[')]))
    
    return _edits



if __name__ == '__main__':
    train_data = read_data('/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep/nopnx/qalb14/train_nopnx.txt')
    context_size = 1
    mle = ContextualTokenClassifier(context_size=context_size)
    mle.train(training_data=train_data)
    dev_data = read_data('/scratch/ba63/arabic-text-editing/gec_data/qalb14/tagger_data/pnx_sep/nopnx/qalb14/dev_nopnx.txt')

    rewritten_data = []
    for example in dev_data:
        subwords = [edit.subword for edit in example]
        predicted_edits = mle.predict(subwords)
        import pdb; pdb.set_trace()
        detok_rewritten_sent, rewritten_sent = rewrite(subwords, predicted_edits)
        rewritten_data.append(detok_rewritten_sent)

    rewritten_data = pnx_tokenize(rewritten_data)
    rewritten_data_nopnx = remove_pnx(rewritten_data)

    # with open(f'dev.mle.{context_size}.txt', mode='w') as f1, open(f'dev.mle.{context_size}.txt.nopnx', mode='w') as f2:
    #     for ex, nopnx_ex in zip(rewritten_data, rewritten_data_nopnx):
    #         f1.write(ex)
    #         f1.write('\n')
    #         f2.write(nopnx_ex)
    #         f2.write('\n')


