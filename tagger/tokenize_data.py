import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from edits.tokenizer import Tokenizer

def read_data(path):
    with open(path) as f:
        return [x.strip() for x in f.readlines()]


def write_data(data, path):
    with open(path, mode='w') as f:
        for example in data:
            for subword in example:
                f.write(subword)
                f.write('\n')
            f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--output')
    parser.add_argument('--tokenizer_path')
    args = parser.parse_args()


    tokenizer = Tokenizer(args.tokenizer_path)
    data = read_data(args.input)
    tokenized_data = [tokenizer.tokenize(example, flatten=True) for example in data]
    write_data(tokenized_data, args.output)





