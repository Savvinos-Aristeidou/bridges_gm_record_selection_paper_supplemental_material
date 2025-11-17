import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize
from scipy import stats


# Define the negative log-likelihood function for the lognormal CDF
def negative_log_likelihood(params, x, empirical_probabilities):
    mu, sigma = params
    # Lognormal CDF
    lognormal_cdf = stats.norm.cdf((np.log(x) - mu) / sigma)
    # Avoid log(0) by clipping probabilities to a minimum value
    lognormal_cdf = np.clip(lognormal_cdf, 1e-10, 1-1e-10)
    # Negative log-likelihood
    nll = -np.sum(empirical_probabilities * np.log(lognormal_cdf) +
                (1 - empirical_probabilities) * np.log(1 - lognormal_cdf))
    return nll


# This is for MSA
def fit_fragilities(SaT1, curvatures, curvatures_grid):
    curvatures[np.isnan(curvatures)] = np.inf
    # Initial guess for mu and sigma
    initial_guess = [np.mean(np.log(SaT1)), np.std(np.log(SaT1))]

    num_of_GMs_per_stripe = curvatures.shape[1]
    mu_hat = np.full(len(curvatures_grid), np.nan)
    sigma_hat = np.full(len(curvatures_grid), np.nan)
    for i, edp in enumerate(curvatures_grid):
        # TODO: check how to treat the collapse cases
        exceedances = np.sum(curvatures > edp, axis=1)
        prob_exceed = exceedances/num_of_GMs_per_stripe
        result = minimize(negative_log_likelihood, initial_guess, args=(SaT1, prob_exceed), method='L-BFGS-B')
        mu_hat[i], sigma_hat[i] = result.x

    curvatures[np.isinf(curvatures)] = np.nan
    return mu_hat, sigma_hat
