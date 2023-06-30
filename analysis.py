"""
Full entropy calculations for PyTorch models
"""
from collections import Counter

import torch.nn as nn
import torch
import transformers
from tqdm import tqdm
import numpy as np

import compression_entropy as ce
from util import log_factorial
import data
import nsb_entropy as ne

def unigram_entropy_direct(tokens: list) -> float:
    n = len(tokens)
    token_n_j = Counter(tokens)
    log_Omega = log_factorial(n) - np.sum([log_factorial(n_j) for n_j in token_n_j.values()])
    return log_Omega / n / np.log(2)


def unigram_entropy_by_counts(tokens: list, token_log_proba: dict, unk_token='') -> float:
    """
    :param tokens: list of tokens that compose a sequence for which we want to compute the entropy
    :param token_log_proba: base-e logarithms of the unigram probability of each token
    :param unk_token: if non-empty, use it for querying the probability distribution for unknown tokens
    :return: the unigram entropy per word
    """
    log_probas = [token_log_proba[token] \
                      if token in token_log_proba or unk_token == '' \
                      else token_log_proba[unk_token] \
                  for token in tokens]
    return -np.mean(log_probas) / np.log(2)


def entropy_rate(model: nn.Module, encodings: torch.Tensor, stride: int, device: str, mask_prompt_tokens: int) -> float:
    """
    Compute the entropy rate (entropy per word) for a given text, given a model and a stride for the sliding window
    :param model: the language model that is used to compute probabilities
    :param encodings: the text for which you want to estimate the entropy, encoded as token IDs
    :param stride: the stride for the sliding window approach. 1 is call LM once per token
    :param device: the device where you want to run the computations (cuda or cpu)
    :param mask_prompt_tokens: number of tokens to mask at the beginning of the entire text (prompt)
    :return: an entropy rate (entropy per word) in bits per word
    """
    max_length = model.config.n_positions
    seq_len = encodings.size(1)

    neg_log_likelihoods = []
    end_loc, prev_end_loc = 0, 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        # Mask prompt tokens
        if begin_loc == 0 and mask_prompt_tokens > 0:
            target_ids[:, :mask_prompt_tokens] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # CrossEntropyLoss averages over input tokens. Multiply it with trg_len to get the summation instead.
            # We will take average over all the tokens in the last step.
            neg_log_likelihood = outputs.loss * trg_len

        neg_log_likelihoods.append(neg_log_likelihood)

        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Division by np.log(2) is change of base to base-2 logarithm
    return (torch.stack(neg_log_likelihoods).sum() / end_loc / np.log(2)).item()

# TODO: warn if the sequence doesn't end in an end-of-sequence token
def full_entropy_calculation(id_text: dict, model: torch.nn.Module,
                             tokenizer: transformers.PreTrainedTokenizerBase,
                             stride: int, device: str,
                             n_prompt_tokens: int,
                             token_log_probas: dict, remove_punctuation: bool,
                             lowercase: bool) -> dict:
    """
    Run a full entropy calculation for some mapping between an ID and a text
    :param id_text: map between some ID (bible, testament, chapter, book, verse) and its corresponding text
    :param model: the model used to compute the probabilities
    :param tokenizer: the tokenizer associated with the model, used for tokenizing the text
    :param stride: for the long-text entropy-calculation algorithm
    :param device: where to run PyTorch calculations
    :param n_prompt_tokens: the number of tokens in the prompt, according to the tokenizer # TODO: get from the tokenizer (but see below)
    :param token_log_probas: base-e-log unigram probability distribution of tokens
    :param remove_punctuation: whether to remove punctuation before computing unigram entropies
    :param lowercase: whether to lowercase before computing unigram entropies
    :return: the entropy rate, the unigram entropy by combinatorics, and the unigram entropy by counts, for each text
    """
    # TODO: add the option to compute the unigram entropy on the encodings
    text_id_entropies = {}
    for text_id, text in id_text.items():
        # Tokenize for the language model
        encodings = tokenizer(text, return_tensors='pt').input_ids.to(device)
        # Tokenize for the unigram entropy computations
        tokens = data.tokenize(text, remove_punctuation, lowercase)
        # Compute the entropy rate
        # TODO: add the option to compute the entropy rate without masking the prompt tokens
        H = entropy_rate(model, encodings, stride, device, n_prompt_tokens)
        # Compute the unigram entropy
        H_s = unigram_entropy_direct(tokens)
        H_r = unigram_entropy_by_counts(tokens, token_log_probas)
        text_id_entropies[text_id] = (H, H_s, H_r)
    return text_id_entropies


def full_entropy_calculation_bpw(id_text: dict,
                                 remove_punct: bool,
                                 lc: bool,
                                 base_name: str) -> dict:
    text_id_entropies = {}
    for text_id, text in id_text.items():
        # Tokenize for the unigram entropy computations
        tokens = data.tokenize(text, remove_punct, lc)
        # Compute the entropy rate
        base_filename = f'{base_name}_{text_id}'
        H = ce.get_entropies_per_word(tokens, base_filename, remove_mismatcher_files=True)
        # Compute the unigram entropy
        H_s = unigram_entropy_direct(tokens)
        token_log_likelihood = data.log_likelihoods(text,
                                                    remove_punctuation=remove_punct,
                                                    lowercase=lc)
        H_r = unigram_entropy_by_counts(tokens, token_log_likelihood)
        text_id_entropies[text_id] = (H, H_s, H_r)
    return text_id_entropies

def get_nsb_entropy(tokens: list) -> float:
    c = Counter(tokens)
    input_histogram = np.array(list(c.values()))
    nsb_entropy = ne.S(ne.make_nxkx(input_histogram, len(c.keys())), input_histogram.sum(),
                       len(c.keys()))
    return float(nsb_entropy)

def nsb_unigram_entropy(id_text: dict,
                        remove_punct: bool,
                        lc: bool) -> dict:
    text_id_entropies = {}
    for text_id, text in id_text.items():
        tokens = data.tokenize(text, remove_punct, lc)
        H_unigram = get_nsb_entropy(tokens)
        text_id_entropies[text_id] = H_unigram
    return text_id_entropies
