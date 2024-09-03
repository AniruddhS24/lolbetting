import numpy as np
import pandas as pd
from datetime import datetime

class NotEnoughDataException(Exception):
    pass

class FeatureExtractor:
    def __init__(self, df, game_desc, window_size=None, values_only=True):
        self.game_desc = game_desc
        self.df_hist = df[df['date'] < game_desc.date]
        self.values_only = values_only
        self.name2func = {
            'feat_assists': lambda: self.average_hist_player('assists'),
            'feat_assistsat10': lambda: self.average_hist_player('assistsat10'),
            'feat_assistsat15': lambda: self.average_hist_player('assistsat15'),
            'feat_assistsat20': lambda: self.average_hist_player('assistsat20'),
            'feat_assistsat25': lambda: self.average_hist_player('assistsat25'),
            'feat_barons': lambda: self.average_hist_player('barons'),
            'feat_ckpm': lambda: self.average_hist_player('ckpm'),
            'feat_csdiffat10': lambda: self.average_hist_player('csdiffat10'),
            'feat_csdiffat15': lambda: self.average_hist_player('csdiffat15'),
            'feat_csdiffat20': lambda: self.average_hist_player('csdiffat20'),
            'feat_csdiffat25': lambda: self.average_hist_player('csdiffat25'),
            'feat_cspm': lambda: self.average_hist_player('cspm'),
            'feat_damageshare': lambda: self.average_hist_player('damageshare'),
            'feat_damagetakenperminute': lambda: self.average_hist_player('damagetakenperminute'),
            'feat_damagetochampions': lambda: self.average_hist_player('damagetochampions'),
            'feat_deaths': lambda: self.average_hist_player('deaths'),
            'feat_deathsat10': lambda: self.average_hist_player('deathsat10'),
            'feat_deathsat15': lambda: self.average_hist_player('deathsat15'),
            'feat_deathsat20': lambda: self.average_hist_player('deathsat20'),
            'feat_deathsat25': lambda: self.average_hist_player('deathsat25'),
            'feat_dpm': lambda: self.average_hist_player('dpm'),
            'feat_dragons': lambda: self.average_hist_player('dragons'),
            'feat_earned_gpm': lambda: self.average_hist_player('earned gpm'),
            'feat_earnedgold': lambda: self.average_hist_player('earnedgold'),
            'feat_earnedgoldshare': lambda: self.average_hist_player('earnedgoldshare'),
            'feat_firstblood': lambda: self.average_hist_player('firstblood'),
            'feat_firstbloodassist': lambda: self.average_hist_player('firstbloodassist'),
            'feat_firstdragon': lambda: self.average_hist_player('firstdragon'),
            'feat_firstherald': lambda: self.average_hist_player('firstherald'),
            'feat_firstmidtower': lambda: self.average_hist_player('firstmidtower'),
            'feat_firsttothreetowers': lambda: self.average_hist_player('firsttothreetowers'),
            'feat_firsttower': lambda: self.average_hist_player('firsttower'),
            'feat_gamelength': lambda: self.average_hist_player('gamelength'),
            'feat_goldat10': lambda: self.average_hist_player('goldat10'),
            'feat_goldat15': lambda: self.average_hist_player('goldat15'),
            'feat_goldat20': lambda: self.average_hist_player('goldat20'),
            'feat_goldat25': lambda: self.average_hist_player('goldat25'),
            'feat_golddiffat10': lambda: self.average_hist_player('golddiffat10'),
            'feat_golddiffat15': lambda: self.average_hist_player('golddiffat15'),
            'feat_golddiffat20': lambda: self.average_hist_player('golddiffat20'),
            'feat_golddiffat25': lambda: self.average_hist_player('golddiffat25'),
            'feat_gpr': lambda: self.average_hist_player('gpr'),
            'feat_gspd': lambda: self.average_hist_player('gspd'),
            'feat_heralds': lambda: self.average_hist_player('heralds'),
            'feat_inhibitors': lambda: self.average_hist_player('inhibitors'),
            'feat_kills': lambda: self.average_hist_player('kills'),
            'feat_killsat10': lambda: self.average_hist_player('killsat10'),
            'feat_killsat15': lambda: self.average_hist_player('killsat15'),
            'feat_killsat20': lambda: self.average_hist_player('killsat20'),
            'feat_killsat25': lambda: self.average_hist_player('killsat25'),
            'feat_result': lambda: self.average_hist_player('result'),
            'feat_team_kpm': lambda: self.average_hist_player('team kpm'),
            'feat_teamdeaths': lambda: self.average_hist_player('teamdeaths'),
            'feat_teamkills': lambda: self.average_hist_player('teamkills'),
            'feat_totalgold': lambda: self.average_hist_player('totalgold'),
            'feat_visionscore': lambda: self.average_hist_player('visionscore'),
            'feat_vspm': lambda: self.average_hist_player('vspm'),
            'feat_xpat25': lambda: self.average_hist_player('xpat25'),
            'feat_xpdiffat25': lambda: self.average_hist_player('xpdiffat25'),
            'feat_opp_assists': lambda: self.average_hist_opponent_gives_up('assists'),
            'feat_opp_assistsat10': lambda: self.average_hist_opponent_gives_up('assistsat10'),
            'feat_opp_assistsat15': lambda: self.average_hist_opponent_gives_up('assistsat15'),
            'feat_opp_assistsat20': lambda: self.average_hist_opponent_gives_up('assistsat20'),
            'feat_opp_assistsat25': lambda: self.average_hist_opponent_gives_up('assistsat25'),
            'feat_opp_barons': lambda: self.average_hist_opponent_gives_up('barons'),
            'feat_opp_ckpm': lambda: self.average_hist_opponent_gives_up('ckpm'),
            'feat_opp_csdiffat10': lambda: self.average_hist_opponent_gives_up('csdiffat10'),
            'feat_opp_csdiffat15': lambda: self.average_hist_opponent_gives_up('csdiffat15'),
            'feat_opp_csdiffat20': lambda: self.average_hist_opponent_gives_up('csdiffat20'),
            'feat_opp_csdiffat25': lambda: self.average_hist_opponent_gives_up('csdiffat25'),
            'feat_opp_cspm': lambda: self.average_hist_opponent_gives_up('cspm'),
            'feat_opp_damageshare': lambda: self.average_hist_opponent_gives_up('damageshare'),
            'feat_opp_damagetakenperminute': lambda: self.average_hist_opponent_gives_up('damagetakenperminute'),
            'feat_opp_damagetochampions': lambda: self.average_hist_opponent_gives_up('damagetochampions'),
            'feat_opp_deaths': lambda: self.average_hist_opponent_gives_up('deaths'),
            'feat_opp_deathsat10': lambda: self.average_hist_opponent_gives_up('deathsat10'),
            'feat_opp_deathsat15': lambda: self.average_hist_opponent_gives_up('deathsat15'),
            'feat_opp_deathsat20': lambda: self.average_hist_opponent_gives_up('deathsat20'),
            'feat_opp_deathsat25': lambda: self.average_hist_opponent_gives_up('deathsat25'),
            'feat_opp_dpm': lambda: self.average_hist_opponent_gives_up('dpm'),
            'feat_opp_dragons': lambda: self.average_hist_opponent_gives_up('dragons'),
            'feat_opp_earned_gpm': lambda: self.average_hist_opponent_gives_up('earned gpm'),
            'feat_opp_earnedgold': lambda: self.average_hist_opponent_gives_up('earnedgold'),
            'feat_opp_earnedgoldshare': lambda: self.average_hist_opponent_gives_up('earnedgoldshare'),
            'feat_opp_firstblood': lambda: self.average_hist_opponent_gives_up('firstblood'),
            'feat_opp_firstbloodassist': lambda: self.average_hist_opponent_gives_up('firstbloodassist'),
            'feat_opp_firstdragon': lambda: self.average_hist_opponent_gives_up('firstdragon'),
            'feat_opp_firstherald': lambda: self.average_hist_opponent_gives_up('firstherald'),
            'feat_opp_firstmidtower': lambda: self.average_hist_opponent_gives_up('firstmidtower'),
            'feat_opp_firsttothreetowers': lambda: self.average_hist_opponent_gives_up('firsttothreetowers'),
            'feat_opp_firsttower': lambda: self.average_hist_opponent_gives_up('firsttower'),
            'feat_opp_gamelength': lambda: self.average_hist_opponent_gives_up('gamelength'),
            'feat_opp_goldat10': lambda: self.average_hist_opponent_gives_up('goldat10'),
            'feat_opp_goldat15': lambda: self.average_hist_opponent_gives_up('goldat15'),
            'feat_opp_goldat20': lambda: self.average_hist_opponent_gives_up('goldat20'),
            'feat_opp_goldat25': lambda: self.average_hist_opponent_gives_up('goldat25'),
            'feat_opp_golddiffat10': lambda: self.average_hist_opponent_gives_up('golddiffat10'),
            'feat_opp_golddiffat15': lambda: self.average_hist_opponent_gives_up('golddiffat15'),
            'feat_opp_golddiffat20': lambda: self.average_hist_opponent_gives_up('golddiffat20'),
            'feat_opp_golddiffat25': lambda: self.average_hist_opponent_gives_up('golddiffat25'),
            'feat_opp_gpr': lambda: self.average_hist_opponent_gives_up('gpr'),
            'feat_opp_gspd': lambda: self.average_hist_opponent_gives_up('gspd'),
            'feat_opp_heralds': lambda: self.average_hist_opponent_gives_up('heralds'),
            'feat_opp_inhibitors': lambda: self.average_hist_opponent_gives_up('inhibitors'),
            'feat_opp_kills': lambda: self.average_hist_opponent_gives_up('kills'),
            'feat_opp_killsat10': lambda: self.average_hist_opponent_gives_up('killsat10'),
            'feat_opp_killsat15': lambda: self.average_hist_opponent_gives_up('killsat15'),
            'feat_opp_killsat20': lambda: self.average_hist_opponent_gives_up('killsat20'),
            'feat_opp_killsat25': lambda: self.average_hist_opponent_gives_up('killsat25'),
            'feat_opp_result': lambda: self.average_hist_opponent_gives_up('result'),
            'feat_opp_team_kpm': lambda: self.average_hist_opponent_gives_up('team kpm'),
            'feat_opp_teamdeaths': lambda: self.average_hist_opponent_gives_up('teamdeaths'),
            'feat_opp_teamkills': lambda: self.average_hist_opponent_gives_up('teamkills'),
            'feat_opp_totalgold': lambda: self.average_hist_opponent_gives_up('totalgold'),
            'feat_opp_visionscore': lambda: self.average_hist_opponent_gives_up('visionscore'),
            'feat_opp_vspm': lambda: self.average_hist_opponent_gives_up('vspm'),
            'feat_opp_xpat25': lambda: self.average_hist_opponent_gives_up('xpat25'),
            'feat_opp_xpdiffat25': lambda: self.average_hist_opponent_gives_up('xpdiffat25'),
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

    def average_hist_player(self, feature_name):
        # OFFENSE How we do against other people
        player_games = self.df_hist[self.df_hist['playername'] == self.game_desc.playername]
        if len(player_games) < 5:
            raise NotEnoughDataException()
        return player_games[feature_name].dropna().mean()

    def average_hist_opponent_gives_up(self, feature_name):
        # DEFENSE Things which other people do on this opponent (how good can this oppoent defend)
        opp_team_games = self.df_hist[self.df_hist['teamname'] == self.game_desc.opp_teamname]
        game_ids = opp_team_games['gameid'].unique()
        all_games = self.df_hist[self.df_hist['gameid'].isin(game_ids)]
        all_games_fltr = all_games[all_games['teamname'] != self.game_desc.opp_teamname]
        all_games_fltr_pos = all_games_fltr[all_games_fltr['position'] == self.game_desc.position]
        if len(all_games_fltr_pos) < 5:
            raise NotEnoughDataException()
        return all_games_fltr_pos[feature_name].dropna().mean()
    
    def recency_predictor(self, feature_name):
        # TODO: what goes here (same patch? past N patches? games?)
        pass