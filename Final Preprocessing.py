import pandas as pd

news_df = pd.read_csv('./news.csv')
shares_df = pd.read_csv('./shares.csv')

news_df['Date'] = pd.to_datetime(news_df['Date'], format='%d-%m-%Y').dt.strftime('%Y-%m-%d')
shares_df['Date'] = pd.to_datetime(shares_df['Date'], format='%Y-%m-%d').dt.strftime('%Y-%m-%d')

merged_df = pd.merge(shares_df, news_df, on='Date', how='left')

merged_df.sort_values(by='Date', inplace=True)

merged_df.dropna(inplace=True)

merged_df.reset_index(drop=True, inplace=True)
columns_to_drop = [col for col in merged_df.columns if 'Adj_Close' in col or 'Volume' in col]
merged_df = merged_df.drop(columns=columns_to_drop)


merged_df.to_csv('data.csv', index=False)
