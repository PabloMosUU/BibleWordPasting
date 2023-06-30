import sys
import json
from compression_entropy import read_selected_verses, get_entropies
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


def train_tokenizer(verses: list, vocab_size: int) -> Tokenizer:
    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=vocab_size)
    tokenizer.train_from_iterator([' '.join(verse) for verse in verses], trainer)
    return tokenizer


def encode_verses(verse_list: list, tokenizer: Tokenizer) -> list:
    return [tokenizer.encode(' '.join(verse)).tokens.copy() for verse in verse_list]


def create_word_split_sets(id_verses: dict, steps_to_save: set) -> dict:
    """
    Given text from a bible, split it using BPE
    :param id_verses: map book IDs to a (ordered) list of verses in the book, each verse being a list of tokens
    :param steps_to_save: the desired vocabulary size increases to be saved
    :return: dictionary mapping book IDs to (dictionary mapping vocabulary size increase to list of tokens)
    """
    book_id_versions = {}
    for book_id, verses in id_verses.items():
        vocab_size_decreases = sorted(steps_to_save)
        tokens = [el for lis in verses for el in lis]
        n_unique_words = len(set(tokens))
        alphabet_size = len(set(list(' '.join(tokens))))
        increase_tokens = {}
        if 0 in steps_to_save:
            increase_tokens[0] = verses.copy()
            vocab_size_decreases.remove(0)
        for vocab_size_decrease in vocab_size_decreases:
            vocab_size = alphabet_size + n_unique_words - vocab_size_decrease
            if vocab_size <= 0:
                print(f'Warning: ignoring vocab_size_decrease={vocab_size_decrease} as it leads to negative vocab size')
                continue
            tokenizer = train_tokenizer(verses, vocab_size)
            output_tokens = encode_verses(verses, tokenizer)
            increase_tokens[vocab_size_decrease] = output_tokens
        book_id_versions[book_id] = increase_tokens
    return book_id_versions


def run_word_splitting(filename: str,
                       lowercase: bool,
                       remove_mismatcher_files: bool,
                       chosen_books: list,
                       truncate_books: bool,
                       split_steps_to_save: set,
                       output_file_path: str,
                       mismatcher_path: str) -> dict:
    selected_book_verses, char_counter = read_selected_verses(filename,
                                                              lowercase,
                                                              chosen_books,
                                                              truncate_books)
    # Create the split versions using BPE
    book_id_versions = create_word_split_sets(selected_book_verses, split_steps_to_save)

    book_id_entropies = {}
    for book_id, n_pairs_verses in book_id_versions.items():
        print(book_id)
        n_pairs_entropies = {}
        for n_pairs, verse_tokens in n_pairs_verses.items():
            print(n_pairs, end='')
            base_filename = f'{output_file_path}/{filename.split("/")[-1]}_{book_id}_v{n_pairs}'
            n_pairs_entropies[n_pairs] = (get_entropies(verse_tokens,
                                                        base_filename,
                                                        remove_mismatcher_files,
                                                        char_counter,
                                                        mismatcher_path),
                                          len(set([el for lis in verse_tokens for el in lis])))
        book_id_entropies[book_id] = n_pairs_entropies
    return book_id_entropies


if __name__ == '__main__':
    assert len(sys.argv) == 5, \
        f'USAGE: python3 {sys.argv[0]} bible_filename temp_dir output_filename mismatcher_filename'
    bible_filename = sys.argv[1]    # The bible filename
    temp_dir = sys.argv[2]          # The directory where Mismatcher files are saved
    output_filename = sys.argv[3]   # The filename where entropies will be saved
    mismatcher_file = sys.argv[4]   # The filename of the mismatcher jar

    split_steps = set([ell for lis in [list(el) for el in (range(-900, 1000, 100), range(-10000, 11000, 1000))] \
                       for ell in lis])

    book_entropies = {}
    for bid in [40, 41, 42, 43, 44, 66]:
        file_book_entropies = run_word_splitting(bible_filename,
                                                    lowercase=True,
                                                    remove_mismatcher_files=True,
                                                    chosen_books=[bid],
                                                    truncate_books=False,
                                                    split_steps_to_save=split_steps,
                                                    output_file_path=temp_dir,
                                                    mismatcher_path=mismatcher_file)
        if bid not in file_book_entropies:
            print(f'WARNING: skipping book {bid} because it is not in {bible_filename}')
            continue
        book_entropies[bid] = file_book_entropies[bid]

    with open(output_filename, 'w') as fp:
        json.dump(book_entropies, fp)
