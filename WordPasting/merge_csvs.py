import sys
import pandas as pd
import os

def merge(dataframe_dir: str) -> pd.DataFrame:
    all_files = os.listdir(dataframe_dir)
    csv_files = [el for el in all_files if el.startswith('entropies_') and el.endswith('.csv')]
    dfs = []
    for el in csv_files:
        dfs.append(pd.read_csv(f'{dataframe_dir}/{el}'))
        dfs[-1]['filename'] = el
    return pd.concat(dfs)

def reformat(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe['language'] = dataframe['filename'].apply(lambda x: x[10:13])
    dataframe['version'] = dataframe.apply(lambda row: f'{row["filename"][16:].split(".")[0]}_{row["book"]}', 1)
    dataframe.rename(columns={
        'orig': 'ent original', 'shuffled': 'ent shuffled', 'masked': 'ent masked', 'iter_id': 'iteration'
    }, inplace=True)
    dataframe.drop(columns=['book_id', 'D_structure', 'D_order', 'filename', 'book'], inplace=True)
    return dataframe

if __name__ == '__main__':
    assert len(sys.argv) == 2, f'USAGE: python3 {sys.argv[0]} csv_dir'
    csv_dir = sys.argv[1]
    df = merge(csv_dir)
    df = reformat(df)
    df.to_csv(f'{csv_dir}/merged.csv', index=False)
