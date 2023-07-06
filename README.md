# Marginalization over tokenizations

This repository contains code for running our _marginalization over tokenizatons_ algorithm presented in our [ACL'23 paper](http://arxiv.org/abs/2306.17757):

```bibtex
@inproceedings{marginaliation,
    title={Should you marginalize over possible tokenizations?},
    author={Chirkova, Nadezhda and Kruszewski, Germ{\'a}n and Rozen, Jos and Dymetman, Marc},
    booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics},
    year={2023},
}
```

## Dependencies

Code uses the following libraries:
* [PyTorch](https://pytorch.org/)
* [HuggingFace transformers](https://huggingface.co/docs/transformers/installation)
* [HuggingFace datasets](https://huggingface.co/docs/datasets/installation)
* [Numpy](https://numpy.org/install/)
* [Pandas](https://pandas.pydata.org/docs/getting_started/install.html)

```bash
python -m venv marginalization
source marginalization/bin/activate
pip3 install numpy pandas torch transformers datasets 
```

## Quick start

By default, the code supports two model families (`gpt2` and `bigscience/bloom-1b7` / `bigscience/bloom-560m`) and three datasets (`wikitext`, `twitter`, and `flores:<lang>`, see language codes [here](https://github.com/facebookresearch/flores/tree/main/flores200)). The model and data will be downloaded automatically using HuggingFace utilities. You can run evaluation as follows:

```bash
python3 main.py --dataset <dataset> --model <model> --log_file <logfile>
```

The script will evaluate bits-per-character (BPC) according to the common practice (using default tokenization) and then according to the proposed marginalization paradigm (using sampled tokenizations). The detailed log with be saved to `<logfile>` and the summary of the results will be printed. 

You can also print the summary of the results for some given log, as follows:

```bash
python3 parse_logs.py <logfile>
```

For example, you can use this command to check the progress of an ongoing evaluation run.

Our logs are available [here](https://drive.google.com/drive/folders/1e9LvOXhaJucn2bS7nvwE2YHdancPuCGy?usp=sharing).


## Hyperparameters

You can specify algorithm / data hyperparameters by passing arguments to the `main.py` script:
* `--max_length`: max length (in tokens) of concatenated text strings. -1 disables concatenation. _Default_: 800
* `--num_texts`: number of (concatenated) strings to evaluate on. _Default_: 100
* `--num_toks_per_seq`: number of tokenizations to sample for each string (K in eq. (2) in the paper). _Default_: 30
* `--max_block_len`: max. length of blocks which strings will be split into (L in the paper). _Default_: 19
* `--batch_size`: number of tokenizations of a block to score with LM in a single batch. _Default_: 16
* `--max_batches_per_block`: max. number of batches per block to score. The number of scored tokenizations per block (M in the paper) is defined as `batch_size` * `max_batches_per_block`. _Default_: 8

We recommend setting `--max_block_len` to the maximum token length in the default tokenization of the data, when possible. You can check token length distribution by running the following script (add flags `--max_length` and `--num_texts` if you use non-default values for them):
```bash
python3 check_token_lengths_distribution.py --model <model> --dataset <dataset> 
```

## Adding custom models or datasets

If you wish to add you model or dataset, please include them in the `aux.py` file. 
* For a dataset, include it in the `load_data` function:
    * return a list of strings;
* For a model, include it in the `load_model_and_tokenizer` function and return:
    * the model;
    * the tokenizer;
    * the `model_max_length` value (the max. number of tokens in the sequence supported by the model);
    * the `is_new_word` function which determines whether a given token is a beginning of a new word (needed for splitting a string into blocks).

## License

The code is distributed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 license. See [LICENSE](LICENSE) for more information.
