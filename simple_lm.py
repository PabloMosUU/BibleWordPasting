"""
This was copied from reproduce_tutorial.py
The code is adapted to do language modeling instead of part-of-speech tagging
"""
import data
from train import EMBEDDING_DIM, HIDDEN_DIM, prepare_sequence, next_word_target
import torch.nn as nn
import torch
import torch.nn.functional as functional
import numpy as np

N_EPOCHS = 300 # normally you would NOT do 300 epochs, it is toy data
LEARNING_RATE = 0.1

class LSTMLanguageModel(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(LSTMLanguageModel, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to next-word space
        self.hidden2word = nn.Linear(hidden_dim, vocab_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        next_word_space = self.hidden2word(lstm_out.view(len(sentence), -1))
        next_word_scores = functional.log_softmax(next_word_space, dim=1)
        return next_word_scores

def invert_dict(key_val: dict) -> dict:
    if len(set(key_val.values())) != len(key_val):
        raise ValueError('Dictionary contains repeated values and cannot be inverted')
    return {v:k for k,v in key_val.items()}

def get_next_words(scores: torch.Tensor, ix_next_word: dict) -> np.ndarray:
    pred_ixs = scores.max(dim=1).indices.numpy()
    return np.vectorize(lambda ix: ix_next_word[ix])(pred_ixs)

def get_word_index(sequences: list) -> dict:
    """
    Generate a look up word->index dictionary from a corpus
    :param sequences: a list of lists of tokens
    :return: a map from word to index
    """
    word_ix = {}
    # For each words-list (sentence) in the training_data
    for sent in sequences:
        for word in sent:
            if word not in word_ix:  # word has not been assigned an index yet
                word_ix[word] = len(word_ix)  # Assign each word with a unique index
    word_ix[data.UNKNOWN_TOKEN] = len(word_ix)
    word_ix[data.CHUNK_END_TOKEN] = len(word_ix)
    return word_ix

def train_sample_(model: nn.Module, sample: list, word_ix: dict, loss_function, optimizer) -> None:
    # Step 1. Remember that Pytorch accumulates gradients. We need to clear them out before each instance
    model.zero_grad()

    # Step 2. Get our inputs ready for the network, that is, turn them into tensors of word indices.
    sentence_in = prepare_sequence(sample, word_ix)
    targets = prepare_sequence(next_word_target(sample), word_ix)

    # Step 3. Run our forward pass.
    partial_pred_scores = model(sentence_in)

    # Step 4. Compute the loss, gradients, and update the parameters by calling optimizer.step()
    loss = loss_function(partial_pred_scores, targets)
    loss.backward()
    optimizer.step()

def train_(model: nn.Module, corpus: list, word_ix: dict, n_epochs: int, loss_function, optimizer) -> None:
    for epoch in range(n_epochs):
        for training_sentence in corpus:
            train_sample_(model, training_sentence, word_ix, loss_function, optimizer)

def pred_sample(model: nn.Module, sample: list, word_ix: dict, ix_word: dict) -> np.ndarray:
    seq = prepare_sequence(sample, word_ix)
    trained_next_word_scores = model(seq)
    return get_next_words(trained_next_word_scores, ix_word)

def pred(model: nn.Module, corpus: list, word_ix: dict, ix_word: dict) -> list:
    with torch.no_grad():
        return [pred_sample(model, seq, word_ix, ix_word) for seq in corpus]

def print_pred(model: nn.Module, corpus: list, word_ix: dict, ix_word: dict) -> None:
    predictions = pred(model, corpus, word_ix, ix_word)
    for prediction in predictions:
        print(prediction)

def initialize_model(embedding_dim, hidden_dim, words_dim, lr):
    model = LSTMLanguageModel(embedding_dim, hidden_dim, words_dim)
    loss_function = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    return model, loss_function, optimizer

if __name__ == '__main__':
    training_data = [
        "The dog ate the apple",
        "Everybody read that book"
    ]
    training_data = [sent.split() for sent in training_data]
    word_to_ix = get_word_index(training_data)
    ix_to_word = invert_dict(word_to_ix)

    lm, nll_loss, sgd = initialize_model(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), lr=LEARNING_RATE)

    print('Before training:')
    print_pred(lm, training_data, word_to_ix, ix_to_word)

    train_(lm, training_data, word_to_ix, N_EPOCHS, nll_loss, sgd)

    print('After training:')
    print_pred(lm, training_data, word_to_ix, ix_to_word)
