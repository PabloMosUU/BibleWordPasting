import numpy as np
import torch
from torch import nn as nn
from torch.nn.functional import log_softmax

import data
import train
from data import prepare_sequence
from train import LSTMLanguageModel


def _get_next_words(scores: torch.Tensor, ix_next_word: dict) -> np.ndarray:
    pred_ixs = scores.max(dim=1).indices.numpy()
    return np.vectorize(lambda ix: ix_next_word[ix])(pred_ixs)


def _pred_sample(model: nn.Module, sample: list, word_ix: dict, ix_word: dict) -> np.ndarray:
    # Put the model in evaluation mode
    model.eval()

    words = sample.copy()
    for i in range(1, len(sample)):
        # Batching is obligatory with my model
        seq = torch.tensor([prepare_sequence(words, word_ix)], dtype=torch.long)
        original_input_sequence_lengths = torch.tensor([len(seq[0])])
        trained_next_word_scores = model(seq, original_input_sequence_lengths)[0]

        word_i = _get_next_words(trained_next_word_scores, ix_word)[i - 1]
        words[i] = word_i
    return np.array(words)


def pred(model: nn.Module, corpus: list, word_ix: dict, ix_word: dict) -> list:
    with torch.no_grad():
        return [_pred_sample(model, seq, word_ix, ix_word) for seq in corpus]


def print_pred(model: nn.Module, corpus: list, word_ix: dict, ix_word: dict) -> None:
    predictions = pred(model, corpus, word_ix, ix_word)
    for prediction in predictions:
        print(' '.join(prediction))


def _get_next_word_log_probabilities(model: LSTMLanguageModel, sample: list, word_ix: dict, ix_word: dict) -> list:
    # Put the model in evaluation mode
    model.eval()

    words = sample.copy()

    # Batching is obligatory with my model
    seq = torch.tensor([prepare_sequence(words, word_ix)], dtype=torch.long)
    original_input_sequence_lengths = torch.tensor([len(seq[0])])
    trained_next_word_scores = log_softmax(model(seq, original_input_sequence_lengths)[0][-1], dim=0)

    return [(ix_word[i], lp.item()) for i, lp in enumerate(trained_next_word_scores)]


def beam_search_decoder(model: LSTMLanguageModel, seed: list, k: int, length: int) -> list:
    """
    Generate a sequence of text based on some seed sequence
    :param model: a trained language model
    :param seed: a list of words to be used as a seed for generation
    :param k: the width of the beam
    :param length: the desired length of the output sequence. If less than 0, stop after encountering an EOS token
    :return: k lists of words generated by the language model (including the seed text), and their neg-log-likelihoods
    """
    sequences = [[seed.copy(), 0.0]]

    # invert the word-to-index dictionary
    ix_word = train.invert_dict(model.word_index)

    # walk over each step in sequence
    while True:
        all_candidates = list()
        # expand each current candidate
        for i in range(len(sequences)):
            seq, score = sequences[i]

            # Generate outputs for the sequence
            next_word_log_probabilities = _get_next_word_log_probabilities(model, seq, model.word_index, ix_word)

            for next_word, next_word_log_softmax in next_word_log_probabilities:
                candidate = [seq + [next_word], score - next_word_log_softmax]
                all_candidates.append(candidate)
        # order all candidates by score
        ordered = sorted(all_candidates, key=lambda tup: tup[1])
        # select k best
        sequences = ordered[:k]

        # Terminate the search
        if len(sequences[0][0]) - len(seed) == length:
            break
        elif length < 0:
            raise NotImplementedError("We do not handle termination at EOS symbol yet")
    return sequences


if __name__ == '__main__':
    model_name = 'dropout00'
    old_model = LSTMLanguageModel.load(f'output/{model_name}.pth')
    print(beam_search_decoder(old_model, [data.START_OF_VERSE_TOKEN], 3, 10))
    print(beam_search_decoder(old_model, [data.START_OF_VERSE_TOKEN], 1, 10))
    print(pred(old_model, [[data.START_OF_VERSE_TOKEN] * 11], old_model.word_index, train.invert_dict(old_model.word_index)))
