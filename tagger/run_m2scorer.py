import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from m2scorer import m2scorer
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--system_output')
parser.add_argument('--m2_file')
parser.add_argument('--mode', default='all')

args = parser.parse_args()

if args.mode == 'all':
    m2scorer.evaluate(args.system_output, args.m2_file, timeout=30)
else:
    m2scorer.evaluate_single_sentences(args.system_output, args.m2_file, timeout=30)