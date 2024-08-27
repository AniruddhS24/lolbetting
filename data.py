from dataclasses import dataclass
from datetime import datetime
import logging
import pandas as pd
import numpy as np
import os

from features import FeatureExtractor
from labels import LabelExtractor

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Input:
    date: datetime
    playername: str
    teamname: str
    opp_teamname: str
    position: str

def _game_from_df(df, df_row):
    game_rows = df[df['gameid'] == df_row['gameid']]
    teams = game_rows['teamname'].unique()
    if teams[0] == df_row['teamname']:
        opp_teamname = teams[1]
    else:
        opp_teamname = teams[0]
    return Input(
        date=df_row['date'],
        playername=df_row['playername'],
        teamname=df_row['teamname'],
        opp_teamname=opp_teamname,
        position=df_row['position'],
    )

def make_dataset(df, start_date, end_date, features, label, save_path=None, save_freq=1000):
    X = []
    y = []
    mod_df = []
    le = LabelExtractor()
    for i, game in df.iterrows():
        try:
            if start_date > game['date'] or game['date'] > end_date:
                continue
            fe = FeatureExtractor(df, _game_from_df(df, game))
            preds = [fe.extract(f) for f in features]
            if np.isnan(preds).any(axis=0):
                logger.warn(f"MISSING DATA {game['gameid']} {game['playername']} ")
                continue
            
            X.append(preds)
            y.append(le.extract(game, label))
            game_dict = game.to_dict()
            game_dict.update({features[j]: preds[j] for j in range(len(preds))})
            game_dict['label'] = y[-1]
            mod_df.append(game_dict)

            logger.info(f"ADDED {game['gameid']} {game['playername']} ")
            if save_path and i > 0 and i % 1000 == 0:
                checkpoint_path = f'{os.path.splittext(save_path)[0]}_{datetime.now().toisostring()}_{i}.csv'
                logger.info(f"SAVING features to {checkpoint_path}")
                pd.DataFrame(mod_df).to_csv(checkpoint_path)
        except Exception as e:
            logger.error(f"ERROR {e}")
            continue
    pd.DataFrame(mod_df).to_csv(save_path)
    return np.array(X), np.array(y)

def load_dataset(features_path, features, label):
    df = pd.read_csv(features_path)
    X = df[features].values
    Y = df[label].values
    return X, Y

def read_data(path):
    df = pd.read_csv(path)
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%y %H:%M')
    df = df.dropna(subset=['playername', 'date'])
    df.sort_values(by='date', inplace=True)
    return df
