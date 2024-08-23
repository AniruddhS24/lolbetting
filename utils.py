import pandas as pd


def read_league_data(playername, before_date):
    df = pd.read_csv('2024_LoL_esports_match_data_from_OraclesElixir.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df[df['date'] < before_date]
    df = df[df['playername'] == playername]
    # df_reversed = df.iloc[::-1].reset_index(drop=True)
    return df
