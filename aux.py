# Marginalization
# Copyright 2023-present NAVER Corp.
# Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license

def load_data(dataset_name: str):
    """
    returns loaded raw data (list of strings)
    """
    if dataset_name == "wikitext":
        from datasets import load_dataset
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"]
        data = [text for text in data if len(text)]
    elif dataset_name == "twitter":
        from datasets import load_dataset
        data = load_dataset("tweet_eval", "emoji", split="test")["text"]
    elif dataset_name[:7] == "flores:":
        from datasets import load_dataset
        lang = dataset_name[7:] # flores dataset_name template is flores:<lang>
        # lang codes at can be found at
        # https://github.com/facebookresearch/flores/tree/main/flores200
        data = load_dataset("facebook/flores", lang)
        data = [item["sentence"] for item in data["devtest"]]
    else:
        raise ValueError("Unknown dataset: %s"%dataset_name)
        
    return data


def load_model_and_tokenizer(model_name):
    """
    returns loaded Huggingface model and tokenizer, 
    and also model_max_length and is_new_word_fun function (see below)
    """
    if "gpt2" in model_name:
        from transformers import GPT2LMHeadModel, GPT2TokenizerFast
        model = GPT2LMHeadModel.from_pretrained(model_name)
        tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        model_max_length = model.config.n_positions
        is_new_word_fun = is_new_word
    elif "bloom" in model_name:
        from transformers import BloomTokenizerFast, BloomForCausalLM
        model = BloomForCausalLM.from_pretrained(model_name)
        tokenizer = BloomTokenizerFast.from_pretrained(model_name)
        model_max_length = 2048
        is_new_word_fun = is_new_word
    else:
        raise ValueError("Unknown model name: %s"%model_name)
        
    return model, tokenizer, model_max_length, is_new_word_fun


def is_new_word(token: str):
    """
    returns True if the given token begins a new word and False otherwise
    """
    return token[0] in {"Ġ", "Ċ"}
