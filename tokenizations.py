# Marginalization
# Copyright 2023-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import sys

def detect_seqlen_dimension_in_keyvalues(model, tokenizer):
    """
    Helper functon to detect which dimension in past_key_values
    returned by model(input, return_dict=False) corresponds to a sequence dimension.
    This value will be needed in the implementation of our proposal distribution
    for more efficient computation, i.e. indexing past_key_values
    """
    def get_shapes(s):
        tok = tokenizer(s, return_tensors="pt")["input_ids"].to(model.device)
        logits, past_key_values = model(tok, return_dict=False)
        return past_key_values[0][0].shape, tok.shape[1]
    pkv_shape, lng = get_shapes('I like transformers')
    pkv_shape = np.array(pkv_shape)
    if sum(pkv_shape==lng) == 1:
        seqlen_dimension_in_keyvalues = np.where(pkv_shape==lng)[0][0]
    else: # plan B test sequence
        pkv_shape, lng = get_shapes('I like transformers very much')
        seqlen_dimension_in_keyvalues = np.where(pkv_shape==lng)[0][0]
    return seqlen_dimension_in_keyvalues

def create_seqs(texts, tokenizer, 
                   num_texts = 100,
                   max_length = 900):
    """
    tokenizes strings and concatenates them into sequences of length <= max_length
    
    Inputs:
    * texts: input data (list of strings)
    * tokenizer: HuggingFace tokenier
    * num_texts: number of concatenated strings to output (int)
    * max_length: max length (in tokens) of concatenated text strings (int)
      If max_length == -1, no concatenation is conducted
    
    Returns:
    * list of (tokenization, string) pairs
    """
    if max_length == -1: # no concatenation needed
        return [[tokenizer(text, return_tensors="pt")["input_ids"], [text]] \
                for text in texts][:num_texts]
    batches = []
    batch = None
    texts_in_batch = []
    current_len = 0
    for te in texts:
        tokenization1 = tokenizer(te, return_tensors="pt")["input_ids"]
        text1 = te
        tokenization2 = tokenizer("\n\n"+te, return_tensors="pt")["input_ids"]
        text2 = "\n\n"+te
        if current_len + tokenization2.shape[1] > max_length and \
        batch is not None: # None may happen at iter 0
            batches.append([batch, texts_in_batch])
            batch = None
            texts_in_batch = []
            current_len = 0
        if tokenization1.shape[1] > max_length:
            batches.append([tokenization1[:, :max_length], [text1]])
        else:
            if batch is None:
                batch = tokenization1
                current_len = batch.shape[1]
                texts_in_batch.append(text1)
            else:
                batch = torch.cat([batch, tokenization2], dim=1)
                current_len += tokenization2.shape[1]
                texts_in_batch.append(text2)
    if batch is not None:
        batches.append([batch, texts_in_batch])

    return batches[:num_texts]

def process_long_word(word, enc, max_block_len=19):
    """
    Splits long words into blocks with length <= max_block_len
    See Appendix C and Figure 3 for more details.
    
    Inputs:
    * word: a list of strs (list of tokens representing a word)
    * enc: a list of ints (list of token_ids representing a word)
    * max_block_len: int (hyperparameter)
    
    Returns:
    * a list of lists of strs (a list of new blocks, 
      each block represented with its tokens)
    * a list of lists of ints (a list of new blocks,
      each block represented with its token_ids)
    """
    words = []
    encs = []
    current_word = ""
    current_enc = []
    for tok, e in zip(word, enc):
        if len(current_word)+len(tok) > max_block_len and\
           current_word != "": 
           # == "" can happen at atep 0
            words.append(current_word)
            encs.append(current_enc)
            current_word = ""
            current_enc = []
        if len(tok) > max_block_len:
            num_chunks = (len(tok) - 1) // max_block_len + 1
            for chunk in range(num_chunks):
                words.append(tok[chunk*max_block_len:
                               (chunk+1)*max_block_len])
                encs.append([-1])
        else:
            current_word += tok
            current_enc.append(e) 
    if current_word != "":
        words.append(current_word)
        encs.append(current_enc)
    return words, encs


def get_block_category(block, standard_encode, i, blocks, is_new_word_fun):
    """
    Detects a block's category, for logging purposes.
    See Appendix C and Figure 3 for more details.
    """
    l = len(blocks)
    if standard_encode==[-1]:
        return "T2"
    elif is_new_word_fun(block):
        if (i < l-1 and is_new_word_fun(blocks[i+1])) or i == l-1:
            return "T0"
        else:
            return "T1"
    else:
        if i == 0:
            return "T0"
        else:
            return "T1"


def compose_blocks_from_standard_tok(input_ids, 
                                     reverse_vocab,
                                     is_new_word_fun,
                                     max_block_len=19):
    """
    Composes blocks from the standard tokenization of a string.
    The high level intuition is that blocks are words 
    (defined using the is_new_word_fun function)
    but if some words are too long, i.e. longer than max_block_len, 
    they are split into smaller blocks
    by following the strategy described in Appendix C and Figure 3.
    
    Inputs:
    * input_ids: a torch tensor of shape [1, num_tokens], dtype = int
    * reverse_vocab: a dict mapping indices to tokens (strs)
    * is_new_word_fun: a functon determining whether a token 
      is a beginning of a new word. The default function is defined in aux.py
      and follows the simple rule common for HF models: token[0] in {"Ġ", "Ċ"}
    * max_block_len: int
    
    Returns:
    * a list of lists of strs (a list of blocks, 
      each block represented with its tokens)
    * a list of lists of ints (a list of blocks,
      each block represented with its token_ids)
    """
    words = []
    encs = []
    word = []
    enc = []
    for token_id in input_ids[0, 1:]:
        token_id = token_id.item()
        token = reverse_vocab[token_id]
        if is_new_word_fun(token):
            # save previous word
            if word != []: # == [] can happen at step 0
                word_joined = "".join(word)
                if len(word_joined) <= max_block_len:
                    words.append(word_joined)
                    encs.append(enc)
                else:
                    words_, encs_ = process_long_word(word, 
                                                      enc,
                                                      max_block_len)
                    words += words_
                    encs += encs_
            word = []
            enc = []
        word.append(token)
        enc.append(token_id)
    if word != []: 
        word_joined = "".join(word)
        if len(word_joined) <= max_block_len:
            words.append(word_joined)
            encs.append(enc)
        else:
            words_, encs_ = process_long_word(word, enc,
                                              max_block_len)
            words += words_
            encs += encs_
    return words, encs


def print_str(s):
    """
    Helper function to log blocks
    """
    return s.replace("\n", "\\n").replace(";", "<semicolon>")


def print_tokens(ids, reverse_vocab): 
    """
    Helper function to log tokenizations
    """
    return [print_str(reverse_vocab[i]) for i in ids]

def splits(string, vocab):
    """ 
    Helper function to enumerate all tokenizations of a block
    adapted from https://stackoverflow.com/questions/60502874/string-split-into-all-possible-combination """
    if type(vocab) != set:
        vocab = set(vocab)
    m = len(string) - 1
    n = (1 << m)
    res = list()

    for i in range(n):
        last = 0
        current = list()

        for j in range(m):
            if (i & (1 << j)):
                current.append(string[last:j+1])
                last = j + 1

        current.append(string[last:])
        res.append(current)

    filtered_splits = list(filter(lambda l: all(t in vocab for t in l), res))
        
    return filtered_splits

def compute_nlls(seqs, 
                 model, 
                 tokenizer, 
                 device,
                 log_file,
                 is_new_word_fun,
                 max_batches_per_block=8,
                 batch_size=16,
                 max_block_len=19,
                 num_toks_per_seq=30,
                 model_max_length=1024,
                 seqlen_dimension_in_keyvalues=2,
                ):
    """
    A top-level function to process each string in the dataset.
    For each string, the negative log-likelihood is computed 
    according to the standard practice (default tokenization) and also
    according to the marginalization paradigm, using importance sampling.
    
    Inputs:
    * seqs: a list of torch tensors, each of shape [1, num_tokens] with dtype=int
    * model: HuggingFace model
    * tokenizer: HuggingFace tokenizer
    * device: "cuda", "cpu", or torch device
    * log_file: a file object where to print logs
    * is_new_word_fun: a function which inputs a token (and) and
      outputs whether this token begins a new word. See example in aux.py
    * max_batches_per_block: int, see arguments in main.py
    * batch_size: int, see arguments in main.py
    * max_block_len: int, see arguments in main.py
    * num_toks_per_seq: int, see arguments in main.py
    * model_max_length: max sequence length supported by the model
    * seqlen_dimension_in_keyvalues: which dimension in past_key_values
      output by model(input, return_dict=True) corresponds to a sequence dimension
      
    Saves results to the log.
    """

    if log_file is None:
        log_file = sys.stdout

    nlls = []
    trg_lens = []
    extras = []
    for input_ids in seqs:
        input_ids = input_ids.to(device)
        
        nlls_over_tries = []
        trg_lens_over_tries = []
        extras_over_tries = []

        # first, evaluate in a standard way
        average_nll, lng = one_batch_processing_standard(model, 
                                                         tokenizer, 
                                                         input_ids,
                                                         log_file)
        nlls_over_tries.append(average_nll)
        trg_lens_over_tries.append(lng)

        # then, evaluate several times with stochastic tokenization
        for _ in range(num_toks_per_seq):
            average_nll, trg_len, extra = \
                one_batch_processing_toksample(model, 
                                               tokenizer, 
                                               input_ids, 
                                               log_file,
                                               is_new_word_fun,
                                               model_max_length, 
                                               batch_size,
                                               max_batches_per_block,
                                               max_block_len,
                                               seqlen_dimension_in_keyvalues)

            nlls_over_tries.append(average_nll)
            trg_lens_over_tries.append(trg_len)
            extras_over_tries.append(extra)
            
        nlls.append(nlls_over_tries)
        trg_lens.append(trg_lens_over_tries)
        extras.append(extras_over_tries)

        prev_ids = input_ids

    return nlls, trg_lens, extras


def one_batch_processing_standard(model, 
                                  tokenizer, 
                                  input_ids, 
                                  log_file):
    """
    Computes logits and nll for a sequence, following the 
    common practice with using the defailt tokenization
    
    See description of arguments in the compute_nll function
    """
    celoss = nn.CrossEntropyLoss()
    celoss_print = nn.CrossEntropyLoss(reduction="none")
    reverse_vocab = {i:v for v, i in tokenizer.vocab.items()}
    target_ids = input_ids.clone()
    
    with torch.no_grad():
        batch_logits, _ = model(input_ids, return_dict=False)
        batch_logits = batch_logits[:, :-1]
        
    # logging
    print("*****Default tokenization*****", file=log_file)
    nll_elems = celoss_print(batch_logits.reshape(-1, batch_logits.shape[-1]), \
                 target_ids[:, 1:].reshape(-1))
    for tok_id, nll_elem in zip(input_ids[0, 1:].cpu().numpy(), 
                                nll_elems.cpu().numpy()):
        vals = [print_str(reverse_vocab[tok_id]), 
                "%.7f"%(-nll_elem),
                ]
        print(";".join(vals), file=log_file)

    nll = celoss(batch_logits.reshape(-1, batch_logits.shape[-1]), \
                 target_ids[:, 1:].reshape(-1))
    
    return nll.item(), input_ids.shape[1]-1
    

def one_batch_processing_toksample(model, 
                                   tokenizer, 
                                   input_ids, 
                                   log_file,
                                   is_new_word_fun,
                                   model_max_length, 
                                   batch_size=16,
                                   max_batches_per_block=8,
                                   max_block_len=19,
                                   seqlen_dimension_in_keyvalues=2):
    """
    Computes logits and nll for a sequence, following 
    the proposed approach with marginalization over tokenizations
    using importance sampling
    
    See description of arguments in the compute_nll function
    """
    pad_idx = tokenizer.unk_token_id
    celoss = nn.CrossEntropyLoss()
    celoss_elemwise = nn.CrossEntropyLoss(ignore_index=-100, 
                                          reduction="none")
    
    vocab = set(list(tokenizer.get_vocab().keys()))
    reverse_vocab = {i:v for v, i in tokenizer.vocab.items()}  
    
    blocks, encs = compose_blocks_from_standard_tok(input_ids,
                                                  reverse_vocab,
                                                  is_new_word_fun,
                                                  max_block_len)
            
    step_logits, past_key_values = model(input_ids[:, :1], \
                                         past_key_values=None, \
                                         return_dict=False)
    batch_logits = [step_logits]
    target_ids = []
    position = 0
    status = 0
    
    print("*****Sampled tokenization*****", file=log_file)

    with torch.no_grad(): 
        for i, (block, standard_encode) in enumerate(zip(blocks, encs)):
            # 1. Compute all possible tokenizations
            # and select top-M shortest of them (Appendix D in the paper)
            all_toks = splits(block, vocab)
            all_toks = sorted(all_toks, key=len)
            all_toks = [tokenizer.convert_tokens_to_ids(ts) for ts in all_toks]
            num_toks = len(all_toks)
            all_toks = all_toks[:max_batches_per_block*batch_size]
            standard_idx = -1
            for idx, tok in enumerate(all_toks):
                if tok == standard_encode:
                    standard_idx = idx
                    
            # 2. Process tokenizations by batches and estimate their likelihoods
            assert len(all_toks) > 0
            losses_sum_all = None
            all_past_key_values = []
            all_step_logits = []
            for bnum in range((len(all_toks)-1)//batch_size+1):
                selected_toks = all_toks[bnum*batch_size:(bnum+1)*batch_size]
                maxlen_local = max([len(tokseq) for tokseq in selected_toks])
                context = [tokseq+[pad_idx]*(maxlen_local-len(tokseq))\
                          for tokseq in selected_toks]
                context = torch.LongTensor(context).to(model.device)
                targets = context.clone()
                targets[targets==pad_idx] = -100

                new_past_key_values = []
                for key, value in past_key_values:
                    key = key.repeat([context.shape[0], 1, 1, 1])
                    value = value.repeat([context.shape[0], 1, 1, 1])
                    new_past_key_values.append([key, value])

                step_logits, cur_past_key_values = model(context, \
                              past_key_values=new_past_key_values, \
                              return_dict=False)
                all_past_key_values.append(cur_past_key_values)
                all_step_logits.append(step_logits)
            
                # logits for the first token were predicted at the prev step
                # so now we compose them with logits for other tokens in current seq
                logits = torch.concat([batch_logits[-1][:, -1:].\
                                       repeat(step_logits.shape[0], 1, 1), 
                                      step_logits[:, :-1]], dim=1)
                # now we estimate P(each continuation)
                losses = celoss_elemwise(logits.reshape(-1, logits.shape[-1]), \
                         targets.reshape(-1)).reshape(targets.shape)
                
                losses_sum = -losses.sum(dim=1)
                losses_sum_all = losses_sum if losses_sum_all is None else \
                                 torch.concat([losses_sum_all, losses_sum], dim=0)
            # 3. Choose tokenization
            option_probs = F.softmax(losses_sum_all, dim=0).cpu().numpy()
            choise = np.random.choice(np.arange(len(option_probs)), p=option_probs)
            
            # logging
            vals = [print_str(block), 
                    get_block_category(block, 
                                      standard_encode, 
                                      i,
                                      blocks,
                                      is_new_word_fun),
                    str(num_toks),
                    " ".join(print_tokens([tok for tok in all_toks[choise]\
                                           if tok != pad_idx], 
                                          reverse_vocab)),
                    "cst" if choise==standard_idx else "cnon-st",
                    "%.7f"%option_probs[choise],
                    "%.7f"%losses_sum_all[choise],
                    "%.7f"%(losses_sum_all[choise]-np.log(option_probs[choise])),
                    "%d"%(len(all_toks)),
                    ]
            
            for idx, prob in enumerate(option_probs):
                if prob >= 0.01 or idx == standard_idx:
                    vals += [
                        " ".join(print_tokens([tok for tok in all_toks[idx]\
                                               if tok != pad_idx], 
                                             reverse_vocab)),
                        "lst" if idx==standard_idx else "lnon-st",
                        "%.7f"%option_probs[idx],
                       ]

            print(";".join(vals), file=log_file, flush=True)
            
            techline = ";".join([str(v) for v in option_probs]) + "|" + \
                       ";".join([str(len(tt)) for tt in all_toks]) + "|" + \
                       str(choise) + "|" + str(len(block)) + "\n"

            # check if max_length if achieved and exit if yes
            position += len(all_toks[choise])
            if position >= model_max_length:
                break
                status = 1
                  
            # prepare logits and key_values for the next step    
            target_ids += all_toks[choise]
            
            new_past_key_values = []
            bnum = choise // batch_size
            bidx = choise % batch_size
            for pair in all_past_key_values[bnum]:
                key, value = pair
                key = torch.index_select(key[bidx:bidx+1],
                                         seqlen_dimension_in_keyvalues, 
                                         torch.arange(len(target_ids)+1)\
                                                        .to(key.device))
                value = torch.index_select(value[bidx:bidx+1],
                                         seqlen_dimension_in_keyvalues, 
                                         torch.arange(len(target_ids)+1)\
                                                        .to(key.device))
                new_past_key_values.append([key, value])
            past_key_values = new_past_key_values
            
            choise_lng = len(all_toks[choise])
            next_token_logits = all_step_logits[bnum][bidx:bidx+1, :choise_lng]
            batch_logits.append(next_token_logits)
            
        batch_logits = torch.concat(batch_logits, dim=1)
        
        nll = celoss(batch_logits[:, :-1].reshape(-1, batch_logits.shape[-1]), \
            torch.LongTensor(target_ids).to(model.device)).item()
        
        return nll, position, [target_ids, status]



