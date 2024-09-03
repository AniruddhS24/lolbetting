from sklearn import linear_model
from scipy.stats import norm, poisson
import numpy as np
import data
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

class BayesianRegression:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.clf = linear_model.BayesianRidge()
        self.clf.fit(self.X, self.Y)

    def predict(self, x):
        return self.clf.predict(x, return_std=True)

    def hit_percentage(self, x, pp_line):
        mu, sigma = self.predict(x)
        p_value = norm.cdf(pp_line, mu, sigma)[0]
        return p_value

class PoissonRegression:
    def __init__(self, X, Y):
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(X)
        self.Y = Y
        self.clf = linear_model.PoissonRegressor()
        self.clf.fit(self.X, self.Y)

    def predict(self, x):
        return self.clf.predict(self.scaler.transform(x))
    
    def hit_percentage(self, x, pp_line):
        lam = self.predict(x)
        p_value = poisson.cdf(pp_line, lam)[0]
        return p_value
    
    def abs_error(self, lam, labels):
        errors = np.abs(labels - lam)
        return np.sum(errors)

def simulate_poisson(X, Y):
    N, M = X.shape[0], X.shape[1]
    losses = []
    for i in range(1, N):
        model = PoissonRegression(X[:i, :], Y[:i])
        lam = model.predict(X[i, :].reshape(-1, M))
        loss = model.abs_error(lam, Y[i])
        losses.append(loss)
        if i % 1000 == 0:
            print(f'Simulated strategy {i}/{N} MAE: {sum(losses)/i}')
    return losses

# if __name__ == '__main__':
#     with open('./features.txt', 'r') as f:
#         features = f.read().split('\n')
#     X, Y = data.load_dataset(
#         './checkpoints/checkpoint_20240901_234012/features.csv',
#         features)
#     print('\n'.join(features))
#     print(X.shape)
#     losses = simulate_poisson(X, Y)
#     plt.plot(losses, linestyle='-', color='b')
#     plt.title('Backtested MAE')
#     plt.xlabel('Iteration')
#     plt.ylabel('MAE')
#     plt.grid(True)
#     plt.show()
