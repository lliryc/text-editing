import re
import json


class Edit:
    def __init__(self, word, edit):
        self.word = word
        self.edit = edit
    
    @classmethod
    def create(cls, aligned_src_chars, aligned_tgt_chars):
        """
        Given a pair of aligned words at the character level, generate an edit
        that will turn the src word into the target word

        Args:
            aligned_src_chars (list of str): src word chars
            aligned_tgt_chars (list of str): tgt word chars
        
        Returns:
            edit (str): the edit that would transform the src word to tgt word
        """

        aligned_src_word = "".join(aligned_src_chars)

        if aligned_src_chars == aligned_tgt_chars:
            return cls(aligned_src_word, "K")  # Keep whole word
        
        elif aligned_src_chars == [''] and aligned_tgt_chars != ['']:
            return cls(aligned_src_word, f"I_[{''.join(aligned_tgt_chars)}]")  # Insert whole word

        elif aligned_src_chars != [''] and aligned_tgt_chars == ['']:
            return cls(aligned_src_word, "D")  # Delete whole word

        elif is_merge(aligned_src_chars, aligned_tgt_chars):
            return cls(aligned_src_word, "".join(['K' if c != ' ' else 'M' for c in aligned_src_chars]))  # Merge

        else:
            edit = cls._generate_detailed_edit(aligned_src_chars, aligned_tgt_chars)
            return cls(aligned_src_word, edit)

    @staticmethod
    def _generate_detailed_edit(aligned_src_chars, aligned_tgt_chars):
        """Helper method to generate detailed edits for non-trivial cases."""

        edit = []

        for src_chars, tgt_chars in zip(aligned_src_chars, aligned_tgt_chars):

            if src_chars == tgt_chars: # Keep or mark spaces in src_chars if necessary
                edit.append('S' if src_chars == ' ' else 'K' * len(src_chars))

            elif src_chars == ' ' and tgt_chars == '': # Merge
                edit.append('M')

            elif src_chars == ' ' and tgt_chars != '': # Merge and Insert
                edit.append(f'MI_[{tgt_chars}]')

            elif src_chars != '' and tgt_chars == '': # Delete and mark spaces as merges if necessary
                edit.append(''.join(['D' if c != ' ' else 'M' for c in src_chars]))
    
            elif src_chars == '' and tgt_chars != '': # Insert
                edit.append(f'I_[{tgt_chars}]')
                # edit.extend([f'I_[{tgt_c}]' for tgt_c in tgt_chars])

            elif len(src_chars) > len(tgt_chars): # Handle len(src_chars) > len(tgt_chars)
                edit.append(get_edits(src_chars, tgt_chars))

            else: # Replace
                edit.append(Edit._replacments(src_chars, tgt_chars))

        return ''.join(edit)
    
    @staticmethod
    def _replacments(src_chars, tgt_chars):
        """Helper method for replacements"""
        if len(tgt_chars) == 1: # Single char replacement
            return f'R_[{tgt_chars}]'

        elif len(src_chars) == len(tgt_chars): # One-to-one replacement
            edits = []
            for i in range(len(src_chars)):
                if src_chars[i] == ' ':
                    edits.append('M')
                    edits.append(f"I_[{tgt_chars[i]}]")
                else:
                    edits.append(f"R_[{tgt_chars[i]}]")

            return ''.join(edits)

        else:  # Replace each src char and insert remaining tgt chars
            replacements = ''.join([f"R_[{tgt_chars[i]}]" for i in range(len(src_chars))])
            insertions = ''.join([f"I_[{t_char}]" for t_char in tgt_chars[len(src_chars):]])
            return replacements + insertions 

    def apply(self, text):
        pass

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_json_str(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self):
        return {'word': self.word, 'edit': self.edit}

    def __len__(self):
        return len(self.edit)

    @classmethod
    def from_json(cls, contents):
        return cls(**contents)



class SubwordEdit:
    def __init__(self, subword, edit):
        self.subword = subword
        self.edit = edit

    def apply(self, subword):
        # Keep
        if self.edit == 'K':
            return subword

        # Appends
        if self.edit.startswith('KA'):
            return self._apply_append(subword, keep=True)

        if self.edit.startswith('DA'):
            return self._apply_append(subword, keep=False)

        # Handle other char-level edits
        _subword = subword.replace('##', '')
        char_edits = re.findall(r'I_\[.*?\]+|A_\[.*?\]+|R_\[.*?\]+|K\*|D\*|.', self.edit)
        edited_subword = self._apply_char_edits(_subword, char_edits)

        # Handle subwords with prefix "##"
        return '##' + edited_subword if '##' in subword else edited_subword

    def _apply_append(self, subword, keep=True):
        """
        Helper method to handle append edits ('KA' or 'DA').
        """
        ops = re.findall(r'A_\[.*?\]+', self.edit)
        inserts = [re.sub(r'A_\[(.*?)\]', r'\1', op) for op in ops]
        return subword + ' ' + ' '.join(inserts) if keep else ''.join(inserts)

    def _apply_char_edits(self, subword, char_edits):
        """
        Apply character-level edits to the word piece (wp).
        """
        edited_subword = ''
        idx = 0

        for i, char_edit in enumerate(char_edits):
            if char_edit == 'K':  # Keep
                edited_subword += subword[idx]
                idx += 1
            
            elif char_edit == 'D':  # Delete
                idx += 1

            elif char_edit.startswith('I'):  # Insert
                edited_subword += re.sub(r'I_\[(.*?)\]', r'\1', char_edit)

            elif char_edit.startswith('A'):  # Append
                edited_subword += (' ' + re.sub(r'A_\[(.*?)\]', r'\1', char_edit) if i
                           else re.sub(r'A_\[(.*?)\]', r'\1', char_edit) + ' ')

            elif char_edit == 'K*':  # Keep and handle remaining edits
                chars_to_keep = self._apply_keep_star(''.join(subword[idx:]), char_edits, i + 1)
                idx += len(chars_to_keep)  # Adjust the index after applying K*
                edited_subword += chars_to_keep

            elif char_edit == 'D*':
                idx += self._apply_delete_star(''.join(subword[idx:]), char_edits, i + 1)

            elif char_edit.startswith('R'):  # Replace
                edited_subword += re.sub(r'R_\[(.*?)\]', r'\1', char_edit)
                idx += 1

        return edited_subword


    def _apply_keep_star(self, subword, char_edits, edit_idx):
        """
        Handle special case of 'K*' if it appears in the beggining of an edit
        """
        remaining_edits = char_edits[edit_idx:]
        inserts = [x for x in remaining_edits if (x.startswith('I') or x.startswith('A'))]

        if len(inserts) == len(remaining_edits):  # if all inserts, add everything
            return ''.join(subword[:])
            
        else: # if not, then add up to the first non-insert edit
            return ''.join(subword[: -(len(remaining_edits) - len(inserts))])


    def _apply_delete_star(self, subword, char_edits, edit_idx):
        remaining_edits = char_edits[edit_idx:]
        inserts_replaces = [x for x in remaining_edits
                            if (x.startswith('I') or x.startswith('A'))]

        if len(inserts_replaces) == len(remaining_edits):  # if all inserts/replaces, delete everything
            return len(subword)
            
        else: # if not, then delete up to the first K edit
            return len(subword[: -(len(remaining_edits) - len(inserts_replaces))])
        


    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_json_str(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self):
        return {'subword': self.subword, 'edit': self.edit}

    @classmethod
    def from_json(cls, contents):
        return cls(**contents)


class Edits:
    """
    A wrapper class to create words edits given an aligned_src_word and its word-level edit
    """
    def __init__(self, words, edits):
        self.words = words
        self.edits = edits

    @classmethod
    def create(cls, aligned_src_word, edit):
        """
        Creates word edits using the word-level src alignment
        and project the char-level edit on the words

        Args:
            aligned_src_word (str): aligned src
            edit (str): char-level edit
        """
        # flatten subwords
        words = [word for word in aligned_src_word.split()]

        if len(words) == 0 and edit.startswith('I'):
            # return cls(subwords, [SubwordEdit('', compress_edit(edit))])
            return cls(words, [SubwordEdit('', compress_edit_new(edit))])
            # return cls(subwords, [SubwordEdit('', edit)])

        if edit == 'K':
            # return cls(subwords, [SubwordEdit(subword, 'K*') for subword in subwords])
            return cls(words, [SubwordEdit(word, 'K') for word in words])

        word_edits = SubwordEdits._project_edit(words, edit)

        # removing extra spaces from the subwords
        words = [word for word in words if word != ' ']

        assert len(words) == len(word_edits)
        return cls(words, word_edits)

    @staticmethod
    def _project_edit(words, edit):
        idx = 0
        word_edits = []
        edit_ops = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|D+|K+|.', edit)
        inserts = [op for op in edit_ops if op.startswith('I_[')]
        replaces = [op for op in edit_ops if op.startswith('R_[')]

        # projecting the edit onto the subwords
        for word in words:
            word_len = len(word)
            word_edit = ''

            while word_len > 0:
                if idx >= len(edit):
                    import pdb; pdb.set_trace()

                if edit[idx] == 'S': # Assign current edit to previous subword in case of S
                    word_edits[-1] += word_edit
                    word_edit = ''
                    idx += 1
                    continue

                if edit[idx] == 'I': # inserts
                    op = inserts.pop(0)
                    word_edit += op
                    idx += len(op)
    
                elif edit[idx] == 'R':
                    op = replaces.pop(0)
                    word_edit += op
                    idx += len(op)
                    word_len -= 1

                elif edit[idx] == 'M': # merges
                    # ensure merges happen first
                    if len(word_edit) != 0:
                        word_edits[-1] = word_edits[-1] + word_edit
                    word_edit = edit[idx]
                    idx += 1

                else: # keeps/deletes
                    if edit[idx] not in ['K', 'D']:
                        import pdb; pdb.set_trace()
                    word_edit += edit[idx]
                    idx += 1
                    word_len -= 1

            word_edits.append(word_edit)

        if idx < len(edit):
            word_edits[-1] = word_edits[-1] + edit[idx:]


        assert ''.join(word_edits) == re.sub(r"(?<!\[)S(?!\])", '', edit)

        assert len(word_edits) == len(words)

        # compressing edits
        # subword_edits = [SubwordEdit(subword, compress_edit(edit))
        #                  for subword, edit in zip(subwords, subword_edits)]

        word_edits = [SubwordEdit(word, compress_edit_new(edit))
                         for word, edit in zip(words, word_edits)]

        # subword_edits = [SubwordEdit(subword, edit) for subword, edit in zip(subwords, subword_edits)]

        return word_edits

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_json_str(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self):
        output = {'words': self.words,
                  'word_edits': [word_edit.edit for word_edit in self.edits]}
        return output
    

class SubwordEdits:
    """
    A wrapper class to create subword edits given an aligned_src_word and its word-level edit
    """
    def __init__(self, subwords, edits):
        self.subwords = subwords
        self.edits = edits

    @classmethod
    def create(cls, aligned_src_word, edit, tokenizer):
        """
        Creates subword edits by tokenizing the word-level src alignment
        and project the char-level edit on the subwords

        Args:
            aligned_src_word (str): aligned src
            edit (str): char-level edit
            tokenizer (Tokenizer): extended tokenizer
        """
        subwords = tokenizer.tokenize(aligned_src_word)

        # flatten subwords
        subwords = [wp for _wp in subwords for wp in _wp]

        if len(subwords) == 0 and edit.startswith('I'):
            # return cls(subwords, [SubwordEdit('', compress_edit(edit))])
            return cls(subwords, [SubwordEdit('', compress_edit_new(edit))])
            # return cls(subwords, [SubwordEdit('', edit)])

        if edit == 'K':
            # return cls(subwords, [SubwordEdit(subword, 'K*') for subword in subwords])
            return cls(subwords, [SubwordEdit(subword, 'K') for subword in subwords])

        subword_edits = SubwordEdits._project_edit(subwords, edit)

        # removing extra spaces from the subwords
        subwords = [wp for wp in subwords if wp != ' ']

        assert len(subwords) == len(subword_edits)
        return cls(subwords, subword_edits)

    @staticmethod
    def _project_edit(subwords, edit):
        idx = 0
        subword_edits = []
        edit_ops = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|D+|K+|.', edit)
        inserts = [op for op in edit_ops if op.startswith('I_[')]
        replaces = [op for op in edit_ops if op.startswith('R_[')]

        # projecting the edit onto the subwords
        for subword in subwords:
            subword_len = len(subword.replace('##',''))
            subword_edit = ''

            while subword_len > 0:
                if idx >= len(edit):
                    import pdb; pdb.set_trace()

                if edit[idx] == 'S': # Assign current edit to previous subword in case of S
                    subword_edits[-1] += subword_edit
                    subword_edit = ''
                    idx += 1
                    continue

                if edit[idx] == 'I': # inserts
                    op = inserts.pop(0)
                    subword_edit += op
                    idx += len(op)
    
                elif edit[idx] == 'R':
                    op = replaces.pop(0)
                    subword_edit += op
                    idx += len(op)
                    subword_len -= 1

                elif edit[idx] == 'M': # merges
                    # ensure merges happen first
                    if len(subword_edit) != 0:
                        subword_edits[-1] = subword_edits[-1] + subword_edit
                    subword_edit = edit[idx]
                    idx += 1

                else: # keeps/deletes
                    if edit[idx] not in ['K', 'D']:
                        import pdb; pdb.set_trace()
                    subword_edit += edit[idx]
                    idx += 1
                    subword_len -= 1

            subword_edits.append(subword_edit)

        if idx < len(edit):
            subword_edits[-1] = subword_edits[-1] + edit[idx:]


        assert ''.join(subword_edits) == re.sub(r"(?<!\[)S(?!\])", '', edit)

        assert len(subword_edits) == len(subwords)

        # compressing edits
        # subword_edits = [SubwordEdit(subword, compress_edit(edit))
        #                  for subword, edit in zip(subwords, subword_edits)]

        subword_edits = [SubwordEdit(subword, compress_edit_new(edit))
                         for subword, edit in zip(subwords, subword_edits)]

        # subword_edits = [SubwordEdit(subword, edit) for subword, edit in zip(subwords, subword_edits)]

        return subword_edits

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_json_str(self):
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_dict(self):
        output = {'subwords': self.subwords,
                  'subword_edits': [subword_edit.edit for subword_edit in self.edits]}
        return output


def get_edits(src_chars, tgt_chars):
    if tgt_chars == '':
        # delete all chars in src_chars
        return 'D' * len(src_chars)

    elif tgt_chars in src_chars:
        # keep what appears in the target and delete the rest
        return ''.join(['D' if c not in tgt_chars else 'K' for c in src_chars])
    
    elif len(src_chars.strip()) == 1 and len(tgt_chars) == 1:
        # replace with tgt and add merge
        return ''.join(['M' if c == ' ' else f'R_[{tgt_chars[0]}]' for c in src_chars])
    
    else:
        edit = ''
        i, j = 0, 0
        # replace source chars with targets whenever possible
        while i < len(src_chars) and j < len(tgt_chars):
            if src_chars[i] != ' ':
                edit += f'R_[{tgt_chars[j]}]'
                j += 1
            elif src_chars[i] == ' ': # merge 
                edit += 'M'
            i += 1

        assert j == len(tgt_chars)

        if i < len(src_chars): # delete the rest of the src chars
            return  edit +  ''.join(['D' * len(src_chars[i:])])
        return edit
        # return ''.join(['D' if c != ' ' else 'S' for c in src_chars] + [f'I_[{c}]' for c in tgt_chars])


def is_merge(aligned_src_chars, aligned_tgt_chars):
    return ''.join([c for c in aligned_src_chars if c != ' ']) == ''.join(aligned_tgt_chars)


# def compress_edit(edit):
#     """Compresses edit string by reducing repeated operations"""
#     grouped_edits = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|D+|K+|.', edit)
#     grouped_edits = compress_insertions(grouped_edits) # reducing multiple insertions into one

#     if len(grouped_edits) == 1 and len(set(grouped_edits[0])) == 1: #e.g., KKK -> K, DDDD -> D
#         return grouped_edits[0][0]
    
#     elif grouped_edits[0] == 'K' * len(grouped_edits[0]):
#         if len(grouped_edits) == 2: #e.g., KKKR_[X] -> K*R_[x]
#             return 'K*' + grouped_edits[1]
        
#         elif grouped_edits[1].startswith('R_') or grouped_edits[1] == 'D': #e.g., KKKR_[x]I -> K*R_[x]I, KKKR_[x]DI -> K*R_[x]DI
#             if all(l.startswith('I_') for l in grouped_edits[2:]):
#                 return 'K*' + grouped_edits[1] + ''.join(grouped_edits[2:])

#     return re.sub('K+$', 'K*', ''.join(grouped_edits))



# def compress_edit(edit):
#     """Compresses edit string by reducing repeated operations"""
#     grouped_edits = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|D+|K+|.', edit)
#     grouped_edits = compress_insertions(grouped_edits) # reducing multiple insertions into one


#     if len(grouped_edits) == 1 and len(set(grouped_edits[0])) == 1: #e.g., KKK -> K, DDDD -> D
#         return grouped_edits[0][0]
    
#     elif grouped_edits[0] == 'K' * len(grouped_edits[0]):
#         if len(grouped_edits) == 2: #e.g., KKKR_[X] -> K*R_[x]
#             return 'K*' + grouped_edits[1]
        
#         elif grouped_edits[1].startswith('R_') or grouped_edits[1] == 'D': #e.g., KKKR_[x]I -> K*R_[x]I, KKKR_[x]DI -> K*R_[x]DI
#             if all(l.startswith('I_') for l in grouped_edits[2:]):
#                 return 'K*' + grouped_edits[1] + ''.join(grouped_edits[2:])

#     compressed_edit = compress_keeps_deletes(grouped_edits, compress_k=True)
#     return compressed_edit
    # return re.sub('K+$', 'K*', ''.join(grouped_edits))



def compress_edit_new(edit):
    grouped_edits = re.findall(r'I_\[.*?\]+|R_\[.*?\]+|A_\[.*?\]+|D+|K+|.', edit)
    grouped_edits = compress_insertions(grouped_edits) # reducing multiple insertions into one
    return ''.join(grouped_edits)




    # if len(grouped_edits) == 1 and len(set(grouped_edits[0])) == 1: # KKKK -> K*, DDDD -> D*
    #     return grouped_edits[0][0] + '*'

    # # Handle cases where there are two groups with 'K' and 'D'
    # # compress the longer sequence
    # if len(grouped_edits) == 2:
    #     first, second = list(set(grouped_edits[0]))[0], list(set(grouped_edits[1]))[0]
    #     if (first == 'K' and second == 'D') or (first == 'D' and second == 'K'):
    #         if len(grouped_edits[0]) > len(grouped_edits[1]):
    #             return f'{first}*' + grouped_edits[1]
    #         else:
    #             return grouped_edits[0] + f'{second}*'

    # seen_edits = []
    # compressed_edit = []

    # # compress Ks and Ds greedily from the end of the edit
    # for i in range(len(grouped_edits) - 1, -1, -1):
    #     _edit = grouped_edits[i]
    #     comp_edit = ''

    #     if _edit != 'M' and len(set(_edit)) == 1:
    #         comp_edit = _edit[0]
    #         assert comp_edit in ['K', 'D']
        
    #     if len(seen_edits) <= 1 and comp_edit:
    #         compressed_edit = [f'{comp_edit}*'] + compressed_edit
    #         return ''.join([grouped_edits[x] for x in range(0, i)] + compressed_edit)
        
    #     else:
    #         compressed_edit = [_edit] + compressed_edit
        
    #     if not _edit.startswith('I') and not _edit.startswith('A'):
    #         seen_edits.append(_edit)

    # return ''.join(compressed_edit)




def compress_insertions(edits):
    """Combines consecutive insertions into one."""
    _edits = []
    insertions = ''
    for edit in edits:
        if edit.startswith('I_'):
            insertions += re.sub(r'I_\[(.*?)\]', r'\1', edit)
        else:
            if insertions:
                _edits.append(f'I_[{insertions}]')
                insertions = ''
            _edits.append(edit)

    if insertions:
        _edits.append(f'I_[{insertions}]')
    
    return _edits