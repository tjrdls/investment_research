import pandas as pd
import os
import glob

stock_code = '005930'
filename = 'data/downloads/{}.csv'.format(stock_code)
print('New filename:', filename)
print('Exists:', os.path.exists(filename))

pattern = 'data/downloads/{}_*.csv'.format(stock_code)
existing_files = glob.glob(pattern)
print('Pattern:', pattern)
print('Found files:', existing_files)

if existing_files:
    # financial.csv 제외
    data_files = [f for f in existing_files if 'financial' not in f]
    if data_files:
        latest_file = max(data_files, key=os.path.getmtime)
        print('Latest file:', latest_file)
        df = pd.read_csv(latest_file, index_col=0, parse_dates=True)
        print('Loaded data shape:', df.shape)
        print('Date range:', df.index.min(), 'to', df.index.max())
        print('Columns:', df.columns.tolist())
    else:
        print('No data files found')