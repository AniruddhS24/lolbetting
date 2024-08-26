import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import montecarlo
import labels

class FeatureExtractor:
    def __init__(self, df, game_desc, window_size=None, values_only=True):
        self.game_desc = game_desc
        self.df_hist = df[df['date'] < game_desc.date]
        self.values_only = values_only
        self.name2func = {
            'agt': self.agt,
            'deaths': self.deaths,
            'apg': self.apg,
            'csdiffat10': self.csdiffat10,
            'csdiffat20': self.csdiffat20,
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
        return self.df_hist[self.df_hist['playername'] == self.game_desc.playername]['gamelength'].dropna().mean()
    
    def apg(self):
        return self.df_hist[self.df_hist['playername'] == self.game_desc.playername]['assists'].dropna().mean()

    def deaths(self):
        return self.df_hist[self.df_hist['playername'] == self.game_desc.playername]['deaths'].dropna().mean()

    def csdiffat10(self):
        return self.df_hist[self.df_hist['playername'] == self.game_desc.playername]['csdiffat10'].dropna().mean()
        
    def csdiffat15(self):
        return self.df_hist[self.df_hist['playername'] == self.game_desc.playername]['csdiffat15'].dropna().mean()
        
    def csdiffat20(self):
        return self.df_hist[self.df_hist['playername'] == self.game_desc.playername]['csdiffat20'].dropna().mean()
    
    def opp_gives_up_kills(self):
        opp_team_games = self.df_hist[self.df_hist['teamname'] == self.game_desc.opp_teamname]
        game_ids = opp_team_games['gameid'].unique()
        all_games = self.df_hist[self.df_hist['gameid'].isin(game_ids)]
        all_games_fltr = all_games[all_games['teamname'] != self.game_desc.opp_teamname]
        all_games_fltr_pos = all_games_fltr[all_games_fltr['position'] == self.game_desc.position]
        return all_games_fltr_pos['kills'].dropna().mean()


