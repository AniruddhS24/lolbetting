import utils
import numpy as np
import pandas as pd
from sklearn import linear_model
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm


class BayesianRegression:
    def __init__(self, player_name, stat, game_date):
        self.player_name = player_name
        self.stat_name = stat
        self.game_date = game_date
        self.league_data = utils.read_league_data(player_name, game_date)
        self.X = []
        self.y = []
        self.LAG = 3
        self.features = []
        for col in self.league_data.columns:
            if not self.league_data[col].isna().any() and not isinstance(self.league_data.iloc[0][col], str) and not isinstance(self.league_data.iloc[0][col], pd.Timestamp) and self.league_data.iloc[0][col].dtype in [np.float64, np.int64]:
                self.features.append(col)
        for i, row in self.league_data.iterrows():
            if i < self.LAG:
                continue
            if i+self.LAG >= len(self.league_data)-1:
                break
            predictors = []
            for j in range(1, self.LAG+1):
                for feat in self.features:
                    predictors.append(self.league_data.iloc[i+j][feat])
            self.X.append(predictors)
            self.y.append(row[self.stat_name])
        self.X = np.array(self.X)
        self.y = np.array(self.y)

    def fit(self):
        clf = linear_model.BayesianRidge()
        clf.fit(self.X, self.y)
        return clf

    def predict(self, clf):
        predictors = []
        for j in range(self.LAG):
            for feat in self.features:
                predictors.append(self.league_data.iloc[j][feat])
        predictors = np.array([predictors])
        return clf.predict(predictors, return_std=True)

    def hit_percentage(self, pp_line):
        clf = self.fit()
        mu, sigma = self.predict(clf)
        p_value = norm.cdf(pp_line, mu, sigma)[0]
        return p_value


if __name__ == '__main__':
    player_name = '1Jiang'
    stat = 'kills'
    game_date = datetime(2024, 8, 15)
    pp_line = 1

    model = BayesianRegression(
        player_name, stat, game_date)
    clf = model.fit()
    mu, sigma = model.predict(clf)
    x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
    y = norm.pdf(x, mu, sigma)
    p_value = norm.cdf(pp_line, mu, sigma)[0]
    plt.axvline(pp_line, color='r', linestyle='--',
                label=f'pp_line = {pp_line}')
    plt.text(pp_line, norm.pdf(pp_line, mu, sigma),
             f'CDF = {p_value:.2f}', fontsize=12, verticalalignment='bottom')
    plt.legend()
    plt.plot(x, y)
    plt.title(f'{player_name} {stat} prediction')
    plt.show()
