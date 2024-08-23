import utils
import labels
import features
from datetime import datetime
import numpy as np
from montecarlo import BayesianRegression
from sim_strategy import simulate

def make_data(league_data, feature_funcs, label_func):
    X = []
    y = []
    for i, row in league_data.iterrows():
        if i < 5:
            continue
        predictors = []
        for ff in feature_funcs:
            predictors.extend(ff(league_data.iloc[:i]))
        X.append(predictors)
        y.append(label_func(row))
    return np.array(X), np.array(y)

if __name__ == '__main__':
    player_name = 'Pollu'
    game_date = datetime(2024, 8, 15)
    data = utils.read_league_data(player_name, game_date)
    X, Y = make_data(data, [features.agt, features.apg], labels.kills)
    print(simulate(X, Y))
