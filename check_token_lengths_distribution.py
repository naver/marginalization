# Marginalization
# Copyright 2023-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import sys
from collections import defaultdict
import pandas as pd
import argparse

import aux
import tokenizations

parser = argparse.ArgumentParser(description='Checking token length distribution')
parser.add_argument('--dataset', type=str, required=True, 
                    help="Dataset to evaluate on. "
                    "Supported by default: wikitext, twitter, flores:<lang>. "
                    "You can add support for other datasets in aux.py",
                    )
parser.add_argument('--model', type=str, required=True,
                    help="Evaluated language model. "
                    "Supported by default: gpt2, bigscience/bloom*, "
                    "e.g. bigscience/bloom-1b7. "
                    "You can add support for other models in aux.py"
                   )
parser.add_argument('--max_length', type=int, default=800, 
                    help="Max length (in tokens) of concatenated text strings; " 
                    "-1 means strings are used as is, without concatenation. "
                    "Should not be greater than model_max_length "
                    "which is a separate setting extracted from model")
parser.add_argument('--num_texts', type=int, default=100, 
                   help="Number of (concatenated) strings to evaluate on")
args = parser.parse_args()

data = aux.load_data(args.dataset)
model, tokenizer, model_max_length, is_new_word_fun = \
                                   aux.load_model_and_tokenizer(args.model)
    
seqs = tokenizations.create_seqs(data, tokenizer, 
                                 num_texts=args.num_texts,
                                 max_length=args.max_length)

token_lengths = defaultdict(lambda: 0)
reverse_vocab = {i:v for v, i in tokenizer.vocab.items()}
for s in seqs:
    for t in s[0][0]:
        token_lengths[len(reverse_vocab[t.item()])] += 1
token_lengths = sorted(token_lengths.items(), 
                       key=lambda x: x[0])
print(pd.DataFrame(token_lengths, 
                   columns=["Token length", "# tokens"]).
      to_markdown(index=False))
