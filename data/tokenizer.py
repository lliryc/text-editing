from transformers import AutoTokenizer
import unicodedata

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
        tokens = self._tokenizer.tokenize(word)

        # replace [UNK] subwords with their original
        if '[UNK]' in tokens:
            if len(tokens) == 1:
                return {'subwords': [word], 'raw_subwords': [word]}
            else:
                tokens = self.replace_unk_with_original(word, tokens,
                                                        strip_accents='ARBERTv2' in self._tokenizer.name_or_path)

        # prepend '##' to each subword if necessary
        subwords = []
        for i, token in enumerate(tokens):
            if i > 0 and not token.startswith("##"):
                # Prepend '##' to all tokens after the first one
                subwords.append("##" + token)
            else:
                subwords.append(token)
        # restored tokens
        raw_subwords = self.restore_tokenized_text(word, subwords)
        return {'subwords': subwords, 'raw_subwords': raw_subwords}
        

    def tokenize(self, text, flatten=False):
        """
        Args:
            text (str): input string
        
        Returns:
            list of list of subwords
        """
        tokenized_words = [self.tokenize_word(word) for word in text.split()]
        raw_subwords = ([subword for word in tokenized_words for subword in word['raw_subwords']] if flatten
                        else [word['raw_subwords'] for word in tokenized_words])

        subwords = ([subword for word in tokenized_words for subword in word['subwords']] if flatten
                    else [word['subwords'] for word in tokenized_words])
        return raw_subwords, subwords


    def replace_unk_with_original(self, text, tokens, strip_accents=False):
        replaced_tokens = []
        current_position = 0

        if strip_accents:
            _text = strip_accents_txt(text)
        else:
            _text = text

        for i, token in enumerate(tokens):
            if token == "[UNK]":
                start = current_position

                # Find the position of the next known token
                next_known_token_index = i + 1
                while (next_known_token_index < len(tokens) and tokens[next_known_token_index].startswith("##")
                       and tokens[next_known_token_index] != '[UNK]'):
                    next_known_token_index += 1

                if next_known_token_index < len(tokens):
                    if tokens[next_known_token_index] != '[UNK]':
                        # Estimate the span of `[UNK]` based on the next token's position
                        next_token_text = tokens[next_known_token_index].replace('##', '')
                        next_position = _text.find(next_token_text, start)
                        end = next_position if next_position != -1 else len(text)
                    else:
                        end = start + 1
                else:
                    end = len(text)

                original_substring = text[start:end]
                replaced_tokens.append(original_substring)  # No "##" prefix for `[UNK]`
                current_position = end
            else:
                if i != 0 and not token.startswith('##'):
                    replaced_tokens.append(f'##{token}')
                else:
                    replaced_tokens.append(token)

                current_position += len(token.replace('##', ''))

        return replaced_tokens


    def restore_tokenized_text(self, text, tokens):
        restored_tokens = []
        raw_index = 0  # Pointer to track the current position in the raw text

        for token in tokens:
            # Remove "##" prefix for subword tokens to get the base form
            is_subword = token.startswith("##")
            token_base = token[2:] if is_subword else token

            # Extract the substring from the raw text corresponding to the token's length
            matching_substring = ""
            for char in text[raw_index:]:
                matching_substring += char
                # Stop when the matching substring length matches the token base
                if len(matching_substring) == len(token_base):
                    break

            # Add the restored token (preserving the "##" prefix if present)
            restored_tokens.append(("##" if is_subword else "") + matching_substring)

            # Update the pointer in the raw text
            raw_index += len(matching_substring)
        
        assert len(restored_tokens) == len(tokens)

        detokenized_text = "".join([token[2:] if token.startswith("##") else token for token in restored_tokens])

        if detokenized_text != text:
            import pdb; pdb.set_trace

        return restored_tokens
    

def strip_accents_txt(text):
    text = unicodedata.normalize("NFD", text)
    return "".join(char for char in text if unicodedata.category(char) != "Mn")


# tokenizer = Tokenizer('/scratch/ba63/BERT_models/ARBERTv2')
# sent = 'القاعدة فوبيا_أو فروعها_ فى كل'
# bla = tokenizer.tokenize(sent, flatten=True)
# import pdb; pdb.set_trace()