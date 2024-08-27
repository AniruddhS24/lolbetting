import numpy as np

def mean_error(mu, std, labels):
    errors = np.abs(labels - mu)
    return np.sum(errors)

def log_likelihood(mu, std, labels):
    n = len(labels)
    return -0.5 * n * np.log(2 * np.pi * std**2) - np.sum((labels - mu)**2) / (2 * std**2)

def simulate(model_class, X, Y):
    losses = 0.0
    for i in range(1, len(X)):
        model = model_class(X[:i, :], Y[:i])
        mu, std = model.predict(X[i, :].reshape(-1, 10))
        print(i, np.expm1(mu), np.expm1(Y[i])) 
        losses += mean_error(mu, std, Y[i])
    return losses/len(X)

