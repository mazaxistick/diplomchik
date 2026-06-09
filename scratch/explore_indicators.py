import os
import pandas as pd

dataset_path = r'c:\Users\thrd3\Documents\Финал\Возможный датасет'
files = [f for f in os.listdir(dataset_path) if f.endswith('.csv') and '102' in f]

out_path = r'c:\Users\thrd3\Documents\Финал\diplomchik-main\scratch\indicators_list.txt'

with open(out_path, 'w', encoding='utf-8') as out_f:
    for f in files:
        try:
            # try utf-8 first, then cp1251
            try:
                df = pd.read_csv(os.path.join(dataset_path, f), sep=';', on_bad_lines='skip', low_memory=False, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(os.path.join(dataset_path, f), sep=';', on_bad_lines='skip', low_memory=False, encoding='cp1251')
                
            if 'indicator_name' in df.columns:
                indicators = df['indicator_name'].unique()
                out_f.write(f"File: {f}\n")
                for ind in indicators[:5]:
                    out_f.write(f"  - {ind}\n")
        except Exception as e:
            out_f.write(f"Error reading {f}: {e}\n")
