from compression_entropy import read_selected_verses, get_entropies
import json
import sys
import os

def run(filename: str,
        lowercase: bool,
        remove_mismatcher_files: bool,
        chosen_books: list,
        truncate_books: bool,
        mismatcher_path: str) -> dict:
    """
    Main program to run the entire pipeline on a single bible
    :param filename: the file containing the bible text
    :param lowercase: whether to lowercase the text before processing
    :param remove_mismatcher_files: whether mismatcher files should be deleted after processing
    :param chosen_books: the books for which you want to compute the entropy (PBC IDs)
    :param truncate_books: whether longer books should be truncated to the length of the shortest
    :param mismatcher_path: full path to the mismatcher executable
    :return: a dictionary with entropy versions and entropies, keyed by book ID
    """
    selected_book_verses, char_counter = read_selected_verses(filename,
                                                              lowercase,
                                                              chosen_books,
                                                              truncate_books)
    # Create a base filename for each book
    book_base_filename = {book_id: 'output/KoplenigEtAl/' + filename.split('/')[-1] + f'_{book_id}' \
                          for book_id in selected_book_verses.keys()}
    return {book_id: get_entropies(verses,
                                   book_base_filename[book_id],
                                   remove_mismatcher_files,
                                   char_counter,
                                   mismatcher_path,
                                   randomize_verses=True) \
            for book_id, verses in selected_book_verses.items()}

if __name__ == '__main__':
    assert len(sys.argv) == 4, \
        f'USAGE: python3 {sys.argv[0]} bibles_dir output_filename mismatcher_filename'
    bibles_dir = sys.argv[1]    # The bible filename
    output_filename = sys.argv[2]   # The filename where entropies will be saved
    mismatcher_file = sys.argv[3]   # The filename of the mismatcher jar

    with open('files_list.txt') as f:
        files_list = f.readlines()

    books = [40, 41, 42, 43, 44, 66]
    entropies = {}
    for bible_filename in files_list:
        entropies[bible_filename] = run(filename=os.path.join(bibles_dir, bible_filename).strip(),
                                        lowercase=True,
                                        remove_mismatcher_files=True,
                                        chosen_books=books,
                                        truncate_books=True,
                                        mismatcher_path=mismatcher_file)

    with open(output_filename, 'w') as fp:
        json.dump(entropies, fp)
