import json
import copy


class InputExample:
    def __init__(self, src_tokens, tgt_tokens):
        self.src_tokens = src_tokens
        self.tgt_tokens = tgt_tokens

    def __repr__(self):
        return str(self.to_json_str())

    def to_json_str(self):
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    def to_dict(self):
        output = copy.deepcopy(self.__dict__)
        return output


class Dataset:
    def __init__(self, raw_data_path):
        self.examples = self.create_examples(raw_data_path)

    def create_examples(self, raw_data_path):

        examples = []
        src_tokens = []
        tgt_tokens = []


        with open(raw_data_path) as f:
            for i, line in enumerate(f.readlines()[1:]):
                line = line.split('\t')
                if len(line) == 2:
                    src, tgt = line
                    if src == '' and tgt != '': # insertions
                        continue

                    src_tokens.append(src.strip())
                    tgt_tokens.append(tgt.strip())

                else:

                    examples.append(InputExample(src_tokens=src_tokens,
                                                 tgt_tokens=tgt_tokens)
                                    )

                    src_tokens, tgt_tokens = [], []


            if src_tokens and tgt_tokens:

                examples.append(InputExample(src_tokens=src_tokens,
                                             tgt_tokens=src_tokens)
                                )
        return examples


    def __getitem__(self, idx):
        return self.examples[idx]


    def __len__(self):
        return len(self.examples)
    

def read_txt_file(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]
    

def write_data(path, data):
    with open(path, mode='w') as f:
        for example in data:
            f.write(example)
            f.write('\n')