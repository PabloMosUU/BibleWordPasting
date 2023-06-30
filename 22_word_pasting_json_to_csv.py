import sys
import pandas as pd
import json

BOOK_ID_NAME = {'40': 'Matthew',
                '41': 'Mark',
                '42': 'Luke',
                '43': 'John',
                '44': 'Acts',
                '66': 'Revelation'}

def rel_error(a):
    assert len(a) == 2
    return abs(a[0] - a[1]) / (a[0] + a[1])

def assert_valid(df: pd.DataFrame) -> None:
    for book_id in df.book_id.unique():
        for iter_id in df.iter_id.unique():
            selection = df[df.apply(lambda row: row['book_id'] == book_id and \
                                                row['iter_id'] == iter_id,
                                    1)]
            if len(selection) == 0:
                continue
            if len(selection) != 1:
                assert len(selection) == 2 and iter_id in ('0', '1000')
                for col in ('orig', 'shuffled', 'masked'):
                    assert rel_error(selection[col].tolist()) * 100 < 0.5
    return

def to_csv(json_file: str) -> None:
    # Read the JSON file
    with open(json_file, 'r') as f:
        book_entropies = json.loads(f.read())
    # Parse the dictionaries into a list of rows, each of which is a dictionary, all with the same keys    rows = []
    row_list = []
    for book_id, version_entropies in book_entropies.items():
        for n_iter, level_entropies in version_entropies.items():
            csv_row = level_entropies.copy()
            csv_row['book_id'] = book_id
            csv_row['iter_id'] = n_iter
            row_list.append(csv_row)
    if not row_list:
        empty_df = pd.DataFrame(columns="orig,shuffled,masked,book_id,iter_id,book,D_structure,D_order".split(','))
        empty_df.to_csv(json_file.replace('.json', '.csv'), index=False)
        return
    # Create a Pandas dataframe
    df = pd.DataFrame(row_list)
    # Perform a check
    assert_valid(df)
    # Map book IDs to their names
    df['book'] = df['book_id'].map(BOOK_ID_NAME)
    # Compute the quantities that are plotted by Koplenig et al
    df['D_structure'] = df.apply(lambda row: row['masked'] - row['orig'], 1)
    df['D_order'] = df.apply(lambda row: row['shuffled'] - row['orig'], 1)
    df.to_csv(json_file.replace('.json', '.csv'), index=False)
    return

if __name__ == '__main__':
    assert len(sys.argv) == 2, f'USAGE: python3 {sys.argv[0]} json_file_name'
    filename = sys.argv[1]
    to_csv(filename)
