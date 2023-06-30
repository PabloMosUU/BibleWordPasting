import csv
import matplotlib.pyplot as plt

def read(filename: str) -> list:
    with open(filename, newline='') as csvfile:
        entropy_reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        rows = list(entropy_reader)
    return rows

def parse(basic_rows: list) -> list:
    header = basic_rows[0]
    body = basic_rows[1:]
    return [{header[i]: el for i, el in enumerate(row)} for row in body]

def plot(data: list, identifier: str, output_dir: str, header: str) -> None:
    """
    Create a plot based on some data
    :param data:
    :param identifier:
    :param output_dir:
    :param header: the name of the column where the iteration ID is stored
    :return:
    """
    unique_books = set([row['book'] for row in data])
    for book_name in unique_books:
        book_data = [row for row in data if row['book'] == book_name]
        x = [float(row['D_order']) for row in book_data]
        y = [float(row['D_structure']) for row in book_data]
        labels = [row[header] for row in book_data]
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        plt.xlabel('Word order information')
        plt.ylabel('Word structure information')
        plt.title(f'{book_name} ({identifier})')
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]), rotation=45)
        plt.savefig(f'{output_dir}/{identifier}_{book_name}.png')
