import util
import sys

if __name__ == '__main__':
    assert len(sys.argv) == 3, f'USAGE: python3 {sys.argv[0]} csv_file_name output_dir'
    csv_filename = sys.argv[1]
    csv_rows = util.read(csv_filename)
    structured_data = util.parse(csv_rows)
    bible_name = csv_filename.split('/')[-1].split('.')[0].split('_')[1].strip()
    bible_plots_dir = sys.argv[2]
    util.plot(structured_data, bible_name, bible_plots_dir)
