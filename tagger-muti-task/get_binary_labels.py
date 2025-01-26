from argparse import ArgumentParser

def convert_labels(input_path, output_path):
    with open(input_path) as f, open(output_path, mode='w') as f2:
        for line in f.readlines():
            line = line.strip()
            if line:
                word, label = line.split('\t')
                if label != 'K*':
                    f2.write(f'{word}\t{label}\tI')
                else:
                    f2.write(f'{word}\t{label}\tC')
                f2.write('\n')

            else:
                f2.write('\n')


parser = ArgumentParser()
parser.add_argument('--input')
parser.add_argument('--output')
args = parser.parse_args()

convert_labels(args.input, args.output)