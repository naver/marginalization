# Marginalization
# Copyright 2023-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import sys
import argparse
import torch

import tokenizations
import aux
import parse_logs

parser = argparse.ArgumentParser(description='Marginalization over tokenizations with importance sampling')
parser.add_argument('--dataset', type=str, required=True, 
                    help="Dataset to evaluate on. "
                    "Supported by default: wikitext, twitter, flores:<lang>. "
                    "You can add support for other datasets in aux.py",
                    )
parser.add_argument('--model', type=str, required=True,
                    help="Evaluated language model. "
                    "Supported by default: gpt2, bigscience/bloom*, "
                    "e.g. bigscience/bloom-1b7. "
                    "You can add support for other models in aux.py")
parser.add_argument('--log_filename', type=str, required=False, default="log.txt",
                    help="Filename where to save logs.")
parser.add_argument('--max_length', type=int, default=800, 
                    help="Max length (in tokens) of concatenated text strings; " 
                    "-1 means strings are used as is, without concatenation. "
                    "Should not be greater than model_max_length "
                    "which is a separate setting extracted from model")
parser.add_argument('--num_texts', type=int, default=100, 
                   help="Number of (concatenated) strings to evaluate on")
parser.add_argument('--num_toks_per_seq', type=int, default=30, 
                   help="Number of tokenizations to sample for each string. "
                    "K in eq. (2) in the paper."
                   )
parser.add_argument('--max_block_len', type=int, default=19, 
                   help="Max. length of blocks which strings will be split into. "
                    "L in the paper. Not recommended to use greater than 21 (slow)")
parser.add_argument('--batch_size', type=int, default=16, 
                   help="Number of tokenizations of a block "
                    "to score with LM in a single batch. "
                    "Recommended to reduce in case of an OOM error, "
                    "along with increasing --max_batches_per_block")
parser.add_argument('--max_batches_per_block', type=int, default=8,
                   help="Max. number of batches per block to score. "
                    "M in the paper (number of scored tokenizations per block)"
                    " is defined as batch_size*max_batches_per_block")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading model...")
model, tokenizer, model_max_length, is_new_word_fun = \
                                   aux.load_model_and_tokenizer(args.model)
model = model.to(device)
print("Done")

print("Loading data...")
data = aux.load_data(args.dataset)
seqlen_dimension_in_keyvalues = \
       tokenizations.detect_seqlen_dimension_in_keyvalues(model, tokenizer)
print("Done")

log_file = open(args.log_filename, "w")
log_file.write("***Command*** "+" ".join(sys.argv)+"\n")

# Data preparation: tokenize strings and concatenate into longer sequences
seqs = tokenizations.create_seqs(data, tokenizer, 
                                 num_texts=args.num_texts,
                                 max_length=args.max_length)

# Run evaluation: evaluates negative log-likelihood for both cases:
# default tokenization and marginalization using importance sampling
# Results are saved into logs
print("Started evaluation...")
nlls_toksam, trg_lens_toksam, qs = tokenizations.compute_nlls(\
                 [seq[0] for seq in seqs], 
                 model, tokenizer, device, log_file,
                 is_new_word_fun=is_new_word_fun,
                 batch_size=args.batch_size,
                 max_batches_per_block=args.max_batches_per_block,
                 max_block_len=args.max_block_len,
                 num_toks_per_seq=args.num_toks_per_seq,
                 model_max_length=model_max_length,
                 seqlen_dimension_in_keyvalues=seqlen_dimension_in_keyvalues,
             )
log_file.close()
print("Done. Logs saved:", args.log_filename)

# Parse logs
all_tabs = parse_logs.parse_log(args.log_filename)
stats = parse_logs.gather_stats(all_tabs)
parse_logs.print_stats(stats, all_tabs)