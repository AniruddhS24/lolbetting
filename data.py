from datetime import datetime
import pandas as pd
import numpy as np

from features import FeatureExtractor, extract_game_metadata

def make_dataset(df, start_date, end_date, features, label_func):
    X = []
    y = []
    for i, game in df.iterrows():
        try:
            if start_date > game['date'] or game['date'] > end_date:
                continue
            fe = FeatureExtractor(df, extract_game_metadata(df, game))
            preds = [fe.extract(f) for f in features]
            if np.isnan(preds).any(axis=0):
                print(f"SKIPPING [{game['date']}] {game['gameid']} player: {game['playername']} ")
                continue
            X.append(preds)
            y.append(label_func(game))
            print(f"ADDED [{game['date']}] {game['gameid']} player: {game['playername']} ")
        except:
            print(f"SKIPPING [{game['date']}] {game['gameid']} player: {game['playername']} ")
            continue
    return np.array(X), np.array(y)

def get_prev_game(playername, date):
    df = read_data()
    game = df.loc[(df['date'] == pd.to_datetime(datetime.strptime(date, '%m/%d/%y %H:%M'))) & (df['playername'] == playername)].iloc[0]
    return df, game

def read_data():
    df = pd.read_csv('2024_LoL_esports_match_data_from_OraclesElixir.csv')
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y %H:%M')
    df = df.dropna(subset=['playername', 'date'])
    df.sort_values(by='date', inplace=True)
    return df
