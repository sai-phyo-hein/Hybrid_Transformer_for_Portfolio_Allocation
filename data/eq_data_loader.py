
import pandas as pd 
import yfinance as yf
import json

def get_eq_data(data_path, start, end, corr_thresh, market_cap_filter):
    """
    A function that select the equities out of nifty50 based on these criteria: 
        - equity is traded during selected date range and no blank
        - all selected equities have mutual correlation less than a threshold
        
    @param data_path: path to a csv file listing all available stocks and related information
    @param start: start date of data to retreive
    @param end: end date of data to retreive
    @param corr_thresh: correlation threshold for stock selection
    @param market_cap_filter: number of stocks to select from top market caps
    @return: dataframe: a dataframe of price data of selected stocks
    
    """
    nf_50_tickers_df = pd.read_csv(data_path)[['Industry', 'Symbol']]
    nf_50_tickers_df.Symbol = nf_50_tickers_df.Symbol + '.NS'
    nf_50_tickers_df['market_cap'] = [yf.Ticker(tick).get_info()['marketCap'] for tick in nf_50_tickers_df.Symbol]

    # get historical data of the selected stocks
    data = yf.download(
        nf_50_tickers_df.Symbol.tolist(), 
        start = start, 
        end = end, 
    )['Adj Close'].dropna(axis = 1)

    # filtering the stocks by the correlation threshold
    corr = data.pct_change().dropna(axis = 0).corr() 
    for i in range(corr.shape[0]): 
        corr.iloc[i, i] = 0.0
    corr_lt_thresh = corr[corr < corr_thresh]
    corr_lt_thresh.dropna(thresh = corr.shape[1], axis = 1, inplace = True)

    # getting top n of stockers from each industry
    selected_tickers = nf_50_tickers_df[
        nf_50_tickers_df.Symbol.isin(corr_lt_thresh.columns)
    ].sort_values(
        'market_cap', ascending = False
    ).groupby('Industry').head(market_cap_filter).Symbol.unique().tolist()

    return data[selected_tickers]

if __name__ == '__main__': 
    pass
