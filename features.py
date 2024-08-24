import numpy as np
import data
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import montecarlo
import labels

def extract_game_metadata(df, game):
    """
    df: The whole excel spreadsheet as a dataframe
    game: A row from the sheet (corresponds to a game)
    """
    # Missing/empty data goes here. If you need extra fields for extracting features, do it here

    # Here I'm finding who the opponent is, for example
    game_rows = df[df['gameid'] == game['gameid']]
    teams = game_rows['teamname'].unique()
    if teams[0] == game['teamname']:
        opp_teamname = teams[1]
    else:
        opp_teamname = teams[0]
    
    # This is what will be self.game_metadata in the feature functions below
    return {
        'date': game['date'],
        'game_number': game['game'],
        'playername': game['playername'],
        'teamname': game['teamname'],
        'opp_teamname': opp_teamname,
        'position': game['position'],
        'champion': game['champion'],
        'patch': game['patch'],
    }

class FeatureExtractor:
    def __init__(self, df, game_metadata, window_size=None, values_only=True):
        self.game_metadata = game_metadata
        self.df_hist = df[df['date'] < game_metadata['date']]
        self.values_only = values_only
        self.name2func = {
            'agt': self.agt,
            'apg': self.apg,
            'csdiffat10': self.csdiffat10,
            'opp_gives_up_kills': self.opp_gives_up_kills,
        }
        self.computed_cache = {}

    def extract(self, feature_name):
        if feature_name in self.computed_cache:
            value = self.computed_cache[feature_name]
        else:
            value = self.name2func[feature_name]()
        if isinstance(value, np.float64):
            value = float(value)
        self.computed_cache[feature_name] = value
        if self.values_only:
            return value
        return (feature_name, value)

    def agt(self):
        return self.df_hist[self.df_hist['playername'] == self.game_metadata['playername']]['gamelength'].dropna().mean()
    
    def apg(self):
        return self.df_hist[self.df_hist['playername'] == self.game_metadata['playername']]['assists'].dropna().mean()

    def deaths(self):
        return self.df_hist[self.df_hist['playername'] == self.game_metadata['playername']]['deaths'].dropna().mean()

    def csdiffat10(self):
        return self.df_hist[self.df_hist['playername'] == self.game_metadata['playername']]['csdiffat10'].dropna().mean()
        
    def csdiffat15(self):
        return self.df_hist[self.df_hist['playername'] == self.game_metadata['playername']]['csdiffat15'].dropna().mean()
        
    def csdiffat20(self):
        return self.df_hist[self.df_hist['playername'] == self.game_metadata['playername']]['csdiffat20'].dropna().mean()
    
    # TODO (neer): Add more functions like this
    def opp_gives_up_kills(self):
        # self.game_metadata gives you info about the game you're making this feature for (i.e. player, team, opponent) --> Future game
        # self.df_hist is a DataFrame of all games before this game (not filtered by player or anything, just excel spreadsheet before this game)
        self.game_metadata = {
            'playername': 'Deft',
            'teamname': 'KT Rolster',
            'position': 'bot',
            'opp_teamname': 'T1'
        }
        opp_team_games = self.df_hist[self.df_hist['teamname'] == self.game_metadata['opp_teamname']]
        game_ids = opp_team_games['gameid'].unique()
        all_games = self.df_hist[self.df_hist['gameid'].isin(game_ids)]
        all_games_fltr = all_games[all_games['teamname'] != self.game_metadata['opp_teamname']]
        all_games_fltr_pos = all_games_fltr[all_games_fltr['position'] == self.game_metadata['position']]
        return all_games_fltr_pos['kills'].dropna().mean()

if __name__ == '__main__':
    # num_games_against
    line = {
        'date': pd.to_datetime(datetime.strptime('8/24/24 12:04', '%m/%d/%y %H:%M')),
        'playername': 'Zeus',
        'teamname': 'T1',
        'opp_teamname': 'KT Rolster',
        'position': 'top',
    }
    features = ['agt', 'apg', 'csdiffat10', 'opp_gives_up_kills']

    df = data.read_data()
    fe = FeatureExtractor(df, line)

    print('Extracting features...')
    X, Y = data.make_dataset(df, datetime(2024, 2, 1), datetime.now(), features, labels.kills)
    print('Fitting Bayesian model...')
    model = montecarlo.BayesianRegression(X, Y)
    print('Running prediction...')
    mu, std = model.predict(np.array([[fe.extract(f) for f in features]]))
    print(f'Predicting {line["playername"]} kills vs {line["opp_teamname"]}')
    print(f'Used features: {features}')
    print(f'mu: {mu[0]}')
    print(f'std: {mu[0]}')
