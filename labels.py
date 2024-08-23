import numpy as np

def kills(league_data_row):
    return np.log1p(league_data_row['kills'])

def winlose(league_data_row):
    return league_data_row['result']