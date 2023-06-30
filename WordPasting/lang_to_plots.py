import os
import util
import sys

def read_language_files(lang: str, input_dir: str) -> list:
    return [f'{input_dir}/{file}' for file in os.listdir(input_dir) \
            if file.startswith(f'entropies_{lang}') and file.endswith('.csv')]

def parse_files(files: list) -> list:
    all_data = [util.parse(util.read(file)) for file in files]
    return [el for lis in all_data for el in lis]

def average_quantities(data: list) -> list:
    unique_n_joins = set([row['iter_id'] for row in data])
    book_name = data[0]['book']
    rows = []
    for n_joins in unique_n_joins:
        n_joins_data = [row for row in data if row['iter_id'] == n_joins]
        H_mean = {'orig': 0, 'shuffled': 0, 'masked': 0}
        for row in n_joins_data:
            for version in H_mean.keys():
                H_mean[version] += float(row[version])
        H_mean = {k: v / len(n_joins_data) for k, v in H_mean.items()}
        H_mean['iter_id'] = n_joins
        H_mean['D_order'] = H_mean['shuffled'] - H_mean['orig']
        H_mean['D_structure'] = H_mean['masked'] - H_mean['orig']
        H_mean['book'] = book_name
        rows.append(H_mean)
    return rows

def run(lang: str, input_dir: str, output_dir: str) -> None:
    # Read all files for that language
    files = read_language_files(lang, input_dir)
    # Create a single data structure with all the files together
    data_frame = parse_files(files)
    # Iterate over books
    unique_books = set([row['book'] for row in data_frame])
    for book_name in unique_books:
        book_data = [row for row in data_frame if row['book'] == book_name]
        # Average the orig, shuffled, and masked quantities
        averaged = average_quantities(book_data)
        # Use this new information to create the plots
        util.plot(averaged, lang, output_dir)
    return

if __name__ == '__main__':
    assert len(sys.argv) == 4, f'USAGE: python3 {sys.argv[0]} language csv_dir output_dir'
    language = sys.argv[1]
    csv_input_dir = sys.argv[2]
    output_plot_dir = sys.argv[3]
    run(language, csv_input_dir, output_plot_dir)
