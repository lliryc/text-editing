from transformers import AutoTokenizer


class Tokenizer:
    def __init__(self, tokenizer):
        self._tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        self._tokenizer._tokenizer.model.max_input_chars_per_word = 1000
    
    def tokenize_word(self, word):
        """
        Tokenizes a single word and inserts ## in case they're dropped
        during pre-tokenization.
        Example: A.B --> ['A', '##.', '##B'] instead of ['A', '.', 'B']

        Args:
            word (str): a single word
        
        Returns:
            subwords (list of str): list of subwords
        """
        # tokenize the input word
        tokens = self._tokenizer.tokenize(word.replace('ګ', 'ك').replace('ي', 'ي'))
        
        # prepend '##' to each subword if necessary
        subwords = []
        for i, token in enumerate(tokens):
            if i > 0 and not token.startswith("##"):
                # Prepend '##' to all tokens after the first one
                subwords.append("##" + token)
            else:
                subwords.append(token)
        
        return subwords
        

    def tokenize(self, text, flatten=False):
        """
        Args:
            text (str): input string
        
        Returns:
            list of list of subwords
        """
        if flatten:
            return [subword for word in text.split() for subword in self.tokenize_word(word)]
        else:
            return [self.tokenize_word(word) for word in text.split()]

