from dataclasses import dataclass
from datetime import datetime
import logging
import pandas as pd
import os
import numpy as np

from features import FeatureExtractor, NotEnoughDataException
from labels import LabelExtractor

logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
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

def compute_features(df, start_date, end_date, label, features=None, save_freq=25000):
    mod_df = []
    le = LabelExtractor()
    ct = 0
    checkpoint_path = os.path.join(os.getcwd(), f"checkpoints/checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(checkpoint_path, exist_ok=True)
    for _, game in df.iterrows():
        try:
            if start_date > game['date'] or game['date'] > end_date:
                continue
            if game['playername'] == 'unknown player':
                logging.warn("Unknown player, skipping...")
                continue
            fe = FeatureExtractor(df, _game_from_df(df, game))
            if not features:
                features = list(fe.name2func.keys())
            try:
                preds = [fe.extract(f) for f in features]
            except NotEnoughDataException:
                logging.warn(f"Skipping. Not enough data for player: {game['playername']} date: {game['date']} gameid: {game['gameid']}")
                continue
            game_dict = game.to_dict()
            game_dict.update({features[j]: preds[j] for j in range(len(preds))})
            game_dict['label'] = le.extract(game, label)
            mod_df.append(game_dict)

            logger.info(f"Extracted features for player: {game['playername']} date: {game['date']} gameid: {game['gameid']}")
            ct += 1
            if ct > 0 and ct % save_freq == 0:
                cpath = os.path.join(checkpoint_path, f"features_{ct}.csv")
                logger.info(f"Saving features to disk: {cpath}")
                pd.DataFrame(mod_df).to_csv(cpath)
                mod_df.clear()
        except Exception as e:
            logger.error(f"{e}")
            continue
    pd.DataFrame(mod_df).to_csv(os.path.join(checkpoint_path, f"features_{ct}.csv"))
    logging.info("Combining checkpoint files...")
    dfs = []
    for checkpt_file in [f for f in os.listdir(checkpoint_path) if f.endswith('.csv')]:
        dfs.append(pd.read_csv(os.path.join(checkpoint_path, checkpt_file)))
    all_features = pd.concat(dfs, ignore_index=True)
    all_features.to_csv(os.path.join(checkpoint_path, f"features.csv"))
    logging.info(f"Saved features master file")

def update_master_features(features_path, features):
    new_features_df = pd.read_csv(features_path)[features + ['label']]
    master_file = "./master_features.csv"
    if os.path.exists(master_file):
        master_df = pd.read_csv(master_file)
    else:
        master_df = pd.DataFrame(columns=features + ['label'])
    updated_master_df = pd.concat([master_df, new_features_df]).drop_duplicates(subset=['gameid', 'playername'], keep='last')
    updated_master_df.to_csv(master_file, index=False)

def load_dataset(features_path, features):
    df = pd.read_csv(features_path)[features + ['label']]
    # print(df.isna().sum().sort_values(ascending=False).head(20))
    df = df.dropna()
    df = df[~np.isinf(df).any(axis=1)]
    X, Y = df[features].values, df['label'].values
    return X, Y 

def get_features(features_path='./features.txt'):
    with open(features_path, 'r') as f:
        features = f.read().split('\n')
    return features

def parse_date(date_str):
    for fmt in ("%m/%d/%y %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return pd.NaT

def read_data(path):
    df = pd.read_csv(path)
    df['date'] = df['date'].apply(parse_date)
    df = df.dropna(subset=['playername', 'date'])
    df.sort_values(by='date', inplace=True)
    return df