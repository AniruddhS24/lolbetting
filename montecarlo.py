import utils
import numpy as np
import pandas as pd
from sklearn import linear_model
from datetime import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm


class BayesianRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.clf = linear_model.BayesianRidge()
        self.clf.fit(self.X, self.Y)

    def predict(self, x):
        return self.clf.predict(x, return_std=True)

    # def hit_percentage(self, pp_line):
    #     clf = self.fit()
    #     mu, sigma = self.predict(clf)
    #     p_value = norm.cdf(pp_line, mu, sigma)[0]
    #     return p_value


# if __name__ == '__main__':
#     player_name = '1Jiang'
#     stat = 'kills'
#     game_date = datetime(2024, 8, 15)
#     pp_line = 1

#     model = BayesianRegression(
#         player_name, stat, game_date)
#     clf = model.fit()
#     mu, sigma = model.predict(clf)
#     x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
#     y = norm.pdf(x, mu, sigma)
#     p_value = norm.cdf(pp_line, mu, sigma)[0]
#     plt.axvline(pp_line, color='r', linestyle='--',
#                 label=f'pp_line = {pp_line}')
#     plt.text(pp_line, norm.pdf(pp_line, mu, sigma),
#              f'CDF = {p_value:.2f}', fontsize=12, verticalalignment='bottom')
#     plt.legend()
#     plt.plot(x, y)
#     plt.title(f'{player_name} {stat} prediction')
#     plt.show()
