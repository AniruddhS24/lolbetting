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

    def hit_percentage(self, pp_line):
        clf = self.fit()
        mu, sigma = self.predict(clf)
        p_value = norm.cdf(pp_line, mu, sigma)[0]
        return p_value
