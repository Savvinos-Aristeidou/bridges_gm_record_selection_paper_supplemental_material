import numpy as np
from scipy.interpolate import interp1d
from scipy import stats

def fragility_function_cdf(x, mu, sigma):
    return stats.norm.cdf((np.log(x) - mu) / sigma)

def get_seismic_demand_hazard_curve(IM, mafes, curvatures_grid, mu_hat, sigma_hat):
    # Create a fine grid for EDP values
    IM_grid = np.linspace(IM.min(), IM.max(), 100)

    # Interpolate hazard curve to get a smooth function
    hazard_curve = interp1d(np.log(IM), np.log(mafes), kind='linear', fill_value="extrapolate")

    annual_rate_of_exceedance = np.zeros_like(curvatures_grid)

    # Calculate the annual rate of exceedance for each EDP level
    for j, edp in enumerate(curvatures_grid):
        p_exceed_all_ims = fragility_function_cdf(IM_grid, mu_hat[j], sigma_hat[j])
        rate_sum = np.trapz(np.exp(hazard_curve(np.log(IM_grid))) * p_exceed_all_ims, np.log(IM_grid))
        annual_rate_of_exceedance[j] = rate_sum
    return annual_rate_of_exceedance