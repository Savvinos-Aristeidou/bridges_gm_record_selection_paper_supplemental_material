from matplotlib import pyplot as plt
from numpy import loadtxt
import  numpy as np
from pathlib import Path
import json
from textwrap import fill

from fit_fragilities import fit_fragilities
from get_seismic_demand_hazard_curve import get_seismic_demand_hazard_curve
from read_oq_hazard_curve import get_hazard_curve_info

cwd = Path.cwd()  # get the current working directory

FONTSIZE = 10
LINEWIDTH = 1.8
MARKERSIZE = 10


model_num = 1

if model_num == 1:
    im_string = ["SA(0.555)", "SA(0.555)", "SA(0.555)", "SA(0.555)", "SA(0.555)", "Sa_avg2(0.555)", "FIV3(0.555)", "Sa_avg2(0.555)", "FIV3(0.555)", "Sa_avg2(0.555)", "FIV3(0.555)"]
    oq_hazard_curve_path = [cwd / 'hazard_curve-mean-SA(0.555)_41.csv',
                            cwd / 'hazard_curve-mean-SA(0.555)_41.csv',
                            cwd / 'hazard_curve-mean-SA(0.555)_41.csv',
                            cwd / 'hazard_curve-mean-SA(0.555)_41.csv',
                            cwd / 'hazard_curve-mean-SA(0.555)_41.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.555)_59.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.555)_53.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.555)_59.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.555)_53.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.555)_59.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.555)_53.csv',
                            ]
    plot_text = "(a) B-1"
elif model_num == 2:
    im_string = ["SA(0.555)", "SA(0.555)", "SA(0.555)", "SA(0.555)", "SA(0.555)", "Sa_avg2(0.555)", "FIV3(0.555)", "Sa_avg2(0.555)", "FIV3(0.555)", "Sa_avg2(0.555)", "FIV3(0.555)"]
    oq_hazard_curve_path = [cwd / 'hazard_curve-mean-SA(0.555)_41.csv',
                            cwd / 'hazard_curve-mean-SA(0.555)_41.csv',
                            cwd / 'hazard_curve-mean-SA(0.555)_41.csv',
                            cwd / 'hazard_curve-mean-SA(0.555)_41.csv',
                            cwd / 'hazard_curve-mean-SA(0.555)_41.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.555)_59.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.555)_53.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.555)_59.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.555)_53.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.555)_59.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.555)_53.csv',
                            ]
    plot_text = "(b) B-2"
elif model_num == 3:
    im_string = ["SA(0.483)", "SA(0.483)", "SA(0.483)", "SA(0.483)", "SA(0.483)", "Sa_avg2(0.483)", "FIV3(0.483)", "Sa_avg2(0.483)", "FIV3(0.483)", "Sa_avg2(0.483)", "FIV3(0.483)"]
    oq_hazard_curve_path = [cwd / 'hazard_curve-mean-SA(0.483)_40.csv',
                            cwd / 'hazard_curve-mean-SA(0.483)_40.csv',
                            cwd / 'hazard_curve-mean-SA(0.483)_40.csv',
                            cwd / 'hazard_curve-mean-SA(0.483)_40.csv',
                            cwd / 'hazard_curve-mean-SA(0.483)_40.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.483)_60.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.483)_54.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.483)_60.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.483)_54.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.483)_60.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.483)_54.csv',
                            ]
    plot_text = "(c) B-3"
elif model_num == 4:
    im_string = ["SA(0.508)", "SA(0.508)", "SA(0.508)", "SA(0.508)", "SA(0.508)", "Sa_avg2(0.508)", "FIV3(0.508)", "Sa_avg2(0.508)", "FIV3(0.508)", "Sa_avg2(0.508)", "FIV3(0.508)"]
    oq_hazard_curve_path = [cwd / 'hazard_curve-mean-SA(0.508)_42.csv',
                            cwd / 'hazard_curve-mean-SA(0.508)_42.csv',
                            cwd / 'hazard_curve-mean-SA(0.508)_42.csv',
                            cwd / 'hazard_curve-mean-SA(0.508)_42.csv',
                            cwd / 'hazard_curve-mean-SA(0.508)_42.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.508)_61.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.508)_55.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.508)_61.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.508)_55.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.508)_61.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.508)_55.csv',
                            ]
    plot_text = "(d) B-4"
elif model_num == 5:
    im_string = ["SA(0.479)", "SA(0.479)", "SA(0.479)", "SA(0.479)", "SA(0.479)", "Sa_avg2(0.479)", "FIV3(0.479)", "Sa_avg2(0.479)", "FIV3(0.479)", "Sa_avg2(0.479)", "FIV3(0.479)"]
    oq_hazard_curve_path = [cwd / 'hazard_curve-mean-SA(0.479)_43.csv',
                            cwd / 'hazard_curve-mean-SA(0.479)_43.csv',
                            cwd / 'hazard_curve-mean-SA(0.479)_43.csv',
                            cwd / 'hazard_curve-mean-SA(0.479)_43.csv',
                            cwd / 'hazard_curve-mean-SA(0.479)_43.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.479)_62.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.479)_56.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.479)_62.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.479)_56.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.479)_62.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.479)_56.csv',
                            ]
    plot_text = "(e) B-5"
elif model_num == 6:
    im_string = ["SA(0.494)", "SA(0.494)", "SA(0.494)", "SA(0.494)", "SA(0.494)", "Sa_avg2(0.494)", "FIV3(0.494)", "Sa_avg2(0.494)", "FIV3(0.494)", "Sa_avg2(0.494)", "FIV3(0.494)"]
    oq_hazard_curve_path = [cwd / 'hazard_curve-mean-SA(0.494)_44.csv',
                            cwd / 'hazard_curve-mean-SA(0.494)_44.csv',
                            cwd / 'hazard_curve-mean-SA(0.494)_44.csv',
                            cwd / 'hazard_curve-mean-SA(0.494)_44.csv',
                            cwd / 'hazard_curve-mean-SA(0.494)_44.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.494)_63.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.494)_57.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.494)_63.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.494)_57.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.494)_63.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.494)_57.csv',
                            ]
    plot_text = "(f) B-6"
elif model_num == 7:
    im_string = ["SA(0.556)", "SA(0.556)", "SA(0.556)", "SA(0.556)", "SA(0.556)", "Sa_avg2(0.556)", "FIV3(0.556)", "Sa_avg2(0.556)", "FIV3(0.556)", "Sa_avg2(0.556)", "FIV3(0.556)"]
    oq_hazard_curve_path = [cwd / 'hazard_curve-mean-SA(0.556)_45.csv',
                            cwd / 'hazard_curve-mean-SA(0.556)_45.csv',
                            cwd / 'hazard_curve-mean-SA(0.556)_45.csv',
                            cwd / 'hazard_curve-mean-SA(0.556)_45.csv',
                            cwd / 'hazard_curve-mean-SA(0.556)_45.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.556)_64.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.556)_58.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.556)_64.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.556)_58.csv',
                            cwd / 'hazard_curve-mean-Sa_avg2(0.556)_64.csv',
                            cwd / 'hazard_curve-mean-FIV3(0.556)_58.csv',
                            ]
    plot_text = "(g) B-7"

poes = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]  # investigation time 50 yrs
poes = np.array(poes)
return_periods = 50/(-np.log(1-poes))

# NOTE: this is hardcoded for bridge 3
GM_set_cases = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

colors = plt.cm.tab20(np.linspace(0, 1, len(GM_set_cases)))  # Tab20 supports up to 20 colors

beta_curvatures_all_cases = np.full((len(poes), len(GM_set_cases)), np.nan)
median_curvatures_all_cases = np.full((len(poes), len(GM_set_cases)), np.nan)
prob_of_collapse_all_cases = np.full((len(poes), len(GM_set_cases)), np.nan)
IMs_MSA = {}
for GM_set_i, GM_set in enumerate(GM_set_cases):
    SaT1 = []
    for iml, poe in enumerate(poes, start=1):
        GM_input_info_path = cwd / "GM_input" / f"Bridge{model_num}" / f"rec_input{GM_set}_IML{iml}.json"

        with open(GM_input_info_path) as f:
            GM_input_info = json.load(f)
        num_of_GMs_per_stripe = len(GM_input_info['selected_scaled_best']['filenames'])
        # TODO: only input the x earthquake for now
        IMi_strings = []
        for key, values in GM_input_info['selected_scaled_best']['IMi'].items():
            if values:
                values = np.array(values, dtype=str)  # Convert values to numpy array as strings
                # IMi_strings.extend(np.char.add(key + "(" + values + ")"))  # Broadcasting with np.char.add for string concatenation
                IMi_strings.extend(list(map(lambda x: f"{key}({x})", map(str, values))))
            else:
                IMi_strings.append(key)
        IMi_strings = np.array(IMi_strings)
        # np.array(GM_input_info['selected_scaled_best']['IMi']['SA']) == 0.483
        SaT1.append(np.array(GM_input_info['selected_scaled_best']['Scaled_IMs'])
                    [:, IMi_strings == im_string[GM_set_i]].mean())

    SaT1 = np.array(SaT1)
    IMs_MSA[GM_set_i] = SaT1
    # IM_string = "Sa1s-Disagg-Select"
    # IM_string = "input2"

    AType = "MSA"

    curvatures = []
    outsdir = f"outs_{AType}_Bridge{model_num}_input{GM_set}"
    for poe in poes:
        logfile = f"{outsdir}/IM_POE_{poe}.txt"

        # read text file into NumPy array
        curvatures.append(loadtxt(logfile))

    curvatures = np.array(curvatures)
    curvatures[curvatures > 0.06] = np.nan  # collapse curvature: 60 mrad

    curvatures_mean = np.exp(np.mean(np.log(curvatures), axis=1))

    beta_curvatures = np.nanstd(np.log(curvatures), axis=1)
    median_curvatures = np.exp(np.nanmean(np.log(curvatures), axis=1))

    prob_of_collapse = np.sum(np.isnan(curvatures), axis=1) / curvatures.shape[1]

    beta_curvatures_all_cases[:, GM_set_i] = beta_curvatures
    median_curvatures_all_cases[:, GM_set_i] = median_curvatures
    prob_of_collapse_all_cases[:, GM_set_i] = prob_of_collapse

    fig, ax = plt.subplots()
    ax.scatter(curvatures * 1000, SaT1.reshape(-1, 1) * np.ones((1, curvatures.shape[1])))
    ax.scatter(curvatures_mean * 1000, SaT1, color='red')
    ax.set_xlabel("Max. peak pier section curvature, $φ_{max}$ [mrad]")
    ax.set_ylabel("$Sa(T_1)$ [g]")

case_codes = ['0: $Sa(T_1)\!-\!$None',
              '1: $Sa(T_1)\!-\!Sa(T)$',
              '2: $Sa(T_1)\!-\!Sa(T), Ds, $\n$Sa_{avg3}, FIV3$',
              '3: $Sa(T_1)\!-\!Sa(T), $\n$Sa_{avg3}, FIV3$',
              '4: $Sa(T_1)\!-\!Sa(T), Ds$',
              '5: $Sa_{avg2}(T_1)\!-\!Sa(T)$',
              '6: $FIV3(T_1)\!-\!Sa(T)$',
              '7: $Sa_{avg2}(T_1)\!-\!Sa_{avg2}(T)$',
              '8: $FIV3(T_1)\!-\!FIV3(T)$',
              '9: $Sa_{avg2}(T_1)\!-\!Sa_{avg2}(T), Ds$',
              '10: $FIV3(T_1)\!-\!FIV3(T), Ds$',]
breakpoint()
fig, ax = plt.subplots(figsize=(4.8, 2.8))
ax.plot(return_periods, median_curvatures_all_cases[:, 0]*1000, marker='o', lw=LINEWIDTH, label="case 0", color=colors[0])
ax.plot(return_periods, median_curvatures_all_cases[:, 1]*1000, marker='o', lw=LINEWIDTH, label="case 1", color=colors[1])
ax.plot(return_periods, median_curvatures_all_cases[:, 2]*1000, marker='o', lw=LINEWIDTH, label="case 2", color=colors[2])
ax.plot(return_periods, median_curvatures_all_cases[:, 3]*1000, marker='o', lw=LINEWIDTH, label="case 3", color=colors[3])
ax.plot(return_periods, median_curvatures_all_cases[:, 4]*1000, marker='o', lw=LINEWIDTH, label="case 4", color=colors[4])
ax.plot(return_periods, median_curvatures_all_cases[:, 5]*1000, marker='o', lw=LINEWIDTH, label="case 5", color=colors[5])
ax.plot(return_periods, median_curvatures_all_cases[:, 6]*1000, marker='o', lw=LINEWIDTH, label="case 6", color=colors[6])
ax.plot(return_periods, median_curvatures_all_cases[:, 7]*1000, marker='o', lw=LINEWIDTH, label="case 7", color=colors[7])
ax.plot(return_periods, median_curvatures_all_cases[:, 8]*1000, marker='o', lw=LINEWIDTH, label="case 8", color=colors[8])
ax.plot(return_periods, median_curvatures_all_cases[:, 9]*1000, marker='o', lw=LINEWIDTH, label="case 9", color=colors[9])
ax.plot(return_periods, median_curvatures_all_cases[:, 10]*1000, marker='o', lw=LINEWIDTH, label="case 10", color=colors[10])
# Add a horizontal dashed line at y=1
# plt.axhline(y=1, color='black', linestyle='--')
plt.grid(True)
plt.xlabel('Return Period, $T_R$ [years]', fontsize=FONTSIZE)
# plt.ylabel('Ratio of medians', fontsize=FONTSIZE)
plt.ylabel('Median, $η_{EDP|IM,NC}$ [mrad]', fontsize=FONTSIZE)
ax.tick_params(axis='both', which='both', labelsize=FONTSIZE)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE+2)
plt.xscale('log')
plt.xticks(return_periods, np.rint(return_periods).astype(int))

# ax.set_xlim([0, None])
ax.set_ylim([0, 60])
# ax.set_ylim([0.8, 1.4])
ax.text(0.05, 0.95, plot_text, transform=ax.transAxes, fontsize=FONTSIZE, 
        fontweight='bold', ha='left', va='top')  # Add text inside subplot

fig.savefig(cwd / "plots" / f"Bridge{model_num}" / f"B{model_num}_median_EDP_given_IML.svg", bbox_inches="tight", format="svg")

fig, ax = plt.subplots(figsize=(4.8, 2.8))
ax.plot(return_periods, prob_of_collapse_all_cases[:, 0]*100, marker='o', lw=LINEWIDTH, label="case 0", color=colors[0])
ax.plot(return_periods, prob_of_collapse_all_cases[:, 1]*100, marker='o', lw=LINEWIDTH, label="case 1", color=colors[1])
ax.plot(return_periods, prob_of_collapse_all_cases[:, 2]*100, marker='o', lw=LINEWIDTH, label="case 2", color=colors[2])
ax.plot(return_periods, prob_of_collapse_all_cases[:, 3]*100, marker='o', lw=LINEWIDTH, label="case 3", color=colors[3])
ax.plot(return_periods, prob_of_collapse_all_cases[:, 4]*100, marker='o', lw=LINEWIDTH, label="case 4", color=colors[4])
ax.plot(return_periods, prob_of_collapse_all_cases[:, 5]*100, marker='o', lw=LINEWIDTH, label="case 5", color=colors[5])
ax.plot(return_periods, prob_of_collapse_all_cases[:, 6]*100, marker='o', lw=LINEWIDTH, label="case 6", color=colors[6])
ax.plot(return_periods, prob_of_collapse_all_cases[:, 7]*100, marker='o', lw=LINEWIDTH, label="case 7", color=colors[7])
ax.plot(return_periods, prob_of_collapse_all_cases[:, 8]*100, marker='o', lw=LINEWIDTH, label="case 8", color=colors[8])
ax.plot(return_periods, prob_of_collapse_all_cases[:, 9]*100, marker='o', lw=LINEWIDTH, label="case 9", color=colors[9])
ax.plot(return_periods, prob_of_collapse_all_cases[:, 10]*100, marker='o', lw=LINEWIDTH, label="case 10", color=colors[10])
# Add a horizontal dashed line at y=1
# plt.axhline(y=1, color='black', linestyle='--')
plt.grid(True)
plt.xlabel('Return Period, $T_R$ [years]', fontsize=FONTSIZE)
# plt.ylabel('Ratio of medians', fontsize=FONTSIZE)
plt.ylabel('Probability of collapse, P(C) [%]', fontsize=FONTSIZE)
ax.tick_params(axis='both', which='both', labelsize=FONTSIZE)
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE+2)
plt.xscale('log')
plt.xticks(return_periods, np.rint(return_periods).astype(int))

# ax.set_xlim([0, None])
ax.set_ylim([0, 100])
# ax.set_ylim([0.8, 1.4])
ax.text(0.05, 0.95, plot_text, transform=ax.transAxes, fontsize=FONTSIZE, 
        fontweight='bold', ha='left', va='top')  # Add text inside subplot

fig.savefig(cwd / "plots" / f"Bridge{model_num}" / f"B{model_num}_prob_of_collapse_given_IML.svg", bbox_inches="tight", format="svg")



# Plotting only the IM levels that impose nonlinearity in the system
fig, ax = plt.subplots(figsize=(4.8, 2.8))
ax.plot(return_periods, beta_curvatures_all_cases[:, 0], marker='o', lw=LINEWIDTH, label="case 0", color=colors[0])
ax.plot(return_periods, beta_curvatures_all_cases[:, 1], marker='o', lw=LINEWIDTH, label="case 1", color=colors[1])
ax.plot(return_periods, beta_curvatures_all_cases[:, 2], marker='o', lw=LINEWIDTH, label="case 2", color=colors[2])
ax.plot(return_periods, beta_curvatures_all_cases[:, 3], marker='o', lw=LINEWIDTH, label="case 3", color=colors[3])
ax.plot(return_periods, beta_curvatures_all_cases[:, 4], marker='o', lw=LINEWIDTH, label="case 4", color=colors[4])
ax.plot(return_periods, beta_curvatures_all_cases[:, 5], marker='o', lw=LINEWIDTH, label="case 5", color=colors[5])
ax.plot(return_periods, beta_curvatures_all_cases[:, 6], marker='o', lw=LINEWIDTH, label="case 6", color=colors[6])
ax.plot(return_periods, beta_curvatures_all_cases[:, 7], marker='o', lw=LINEWIDTH, label="case 7", color=colors[7])
ax.plot(return_periods, beta_curvatures_all_cases[:, 8], marker='o', lw=LINEWIDTH, label="case 8", color=colors[8])
ax.plot(return_periods, beta_curvatures_all_cases[:, 9], marker='o', lw=LINEWIDTH, label="case 9", color=colors[9])
ax.plot(return_periods, beta_curvatures_all_cases[:, 10], marker='o', lw=LINEWIDTH, label="case 10", color=colors[10])
# Add a horizontal dashed line at y=1
# plt.axhline(y=1, color='black', linestyle='--')
plt.grid(True)
plt.xlabel('Return Period, $T_R$ [years]', fontsize=FONTSIZE)
# plt.ylabel('Ratio of betas', fontsize=FONTSIZE)
plt.ylabel('Dispersion, $β_{EDP|IM,NC}$', fontsize=FONTSIZE)
ax.tick_params(axis='both', which='both', labelsize=FONTSIZE)

# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE+2)
plt.xscale('log')
plt.xticks(return_periods, return_periods.astype(int))
# ax.set_xlim([3.9, None])
# ax.set_ylim([0.5, 2.])
ax.set_ylim([0, 1.2])
# ax.set_ylim([0.8, 1.4])
ax.text(0.95, 0.95, plot_text, transform=ax.transAxes, fontsize=FONTSIZE, 
        fontweight='bold', ha='right', va='top')  # Add text inside subplot

fig.savefig(cwd / "plots" / f"Bridge{model_num}" / f"B{model_num}_dispersion_EDP_given_IML.svg", bbox_inches="tight", format="svg")

# Plotting the seismic demand hazard curve
fig, ax = plt.subplots(figsize=(4.8, 2.8))
fig2, ax2 = plt.subplots(figsize=(4.8, 2.8))
fig3, ax3 = plt.subplots(figsize=(4.8, 2.8))

for GM_set_i, GM_set in enumerate(GM_set_cases, start=1):
    IM, mafes = get_hazard_curve_info(oq_hazard_curve_path[GM_set_i-1])
    curvatures = []
    outsdir = f"outs_{AType}_Bridge{model_num}_input{GM_set}"
    for poe in poes:
        logfile = f"{outsdir}/IM_POE_{poe}.txt"

        # read text file into NumPy array
        curvatures.append(loadtxt(logfile))
    curvatures = np.array(curvatures)

    curvatures[curvatures > 0.06] = np.nan  # collapse curvature: 60 mrad
    curvatures_grid = np.linspace(np.nanmin(curvatures), np.nanmax(curvatures), 30)

    mu_hat, sigma_hat = fit_fragilities(IMs_MSA[GM_set_i-1], curvatures, curvatures_grid)

    annual_rate_of_exceedance = get_seismic_demand_hazard_curve(IM, mafes, curvatures_grid, mu_hat, sigma_hat)

    ax.plot(curvatures_grid*1000, annual_rate_of_exceedance, lw=LINEWIDTH+0.3, label=f"case {GM_set_i-1}", color=colors[GM_set_i-1])
    ax2.plot(curvatures_grid*1000, sigma_hat, lw=LINEWIDTH+0.3, label=f"case {GM_set_i-1}", color=colors[GM_set_i-1])
    ax3.plot(curvatures_grid*1000, np.exp(mu_hat), lw=LINEWIDTH+0.3, label=case_codes[GM_set_i-1], color=colors[GM_set_i-1])

ax.set_xlabel('EDP: pier section curvature, $φ_{max}$ [mrad]', fontsize=FONTSIZE+2)
# plt.xlabel('Damage ratio DR [-]')
ax.set_ylabel('Annual Rate of Exceedance', fontsize=FONTSIZE+2)
# ax.set_title('Seismic Demand Hazard Curve', fontsize=FONTSIZE+2)
ax.grid(True)
ax.set_yscale('log')
ax.set_ylim([1e-5, 1e-1])
ax.set_xlim([0, 60])
# ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE+2)
ax.tick_params(axis='both', which='both', labelsize=FONTSIZE+2)
ax.text(0.95, 0.95, plot_text, transform=ax.transAxes, fontsize=FONTSIZE+2, 
        fontweight='bold', ha='right', va='top')  # Add text inside subplot

fig.savefig(cwd / "plots" / f"Bridge{model_num}" / f"B{model_num}_seismic_demand_hazard_curve.svg", bbox_inches="tight", format="svg")
# TODO: seismic demand hazard curves should start from zero EDP, and MAFE should take the value of the baseline hazard, which is the hazard curve with an infinitesimally small intensity 

ax2.set_xlabel('EDP: pier section curvature, $φ_{max}$ [mrad]', fontsize=FONTSIZE+2)
# plt.xlabel('Damage ratio DR [-]')
ax2.set_ylabel('Dispersion, $β_{IM|EDP}$', fontsize=FONTSIZE+2)
# ax2.axvline(x=1.25, lw=LINEWIDTH+0.3, color='black')  # yielding curvature
ax2.grid(True)
ax2.set_ylim([0, 1])
ax2.set_xlim([0, 60])
# ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE+2)
ax2.text(0.5, 0.95, plot_text, transform=ax2.transAxes, fontsize=FONTSIZE+2, 
        fontweight='bold', ha='center', va='top')  # Add text inside subplot

fig2.savefig(cwd / "plots" / f"Bridge{model_num}" / f"B{model_num}_dispersion_IM_given_edp_exceedance_of_EDP.svg", bbox_inches="tight", format="svg")

ax3.set_xlabel('EDP: pier section curvature, $φ_{max}$ [mrad]', fontsize=FONTSIZE+2)
# plt.xlabel('Damage ratio DR [-]')
ax3.set_ylabel('Median, $η_{IM|EDP}$', fontsize=FONTSIZE+2)
ax3.grid(True)
ax3.set_ylim([0, 5])
ax3.set_xlim([0, 60])
# ax3.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=FONTSIZE+2)
ax3.text(0.05, 0.95, plot_text, transform=ax3.transAxes, fontsize=FONTSIZE+2, 
        fontweight='bold', ha='left', va='top')  # Add text inside subplot

fig3.savefig(cwd / "plots" / f"Bridge{model_num}" / f"B{model_num}_median_IM_given_edp_exceedance_of_EDP.svg", bbox_inches="tight", format="svg")

fig4, ax4 = plt.subplots(figsize=(4.8, 2.8))
# Extract legend handles and labels
handles, labels = ax3.get_legend_handles_labels()
for handle in handles:
    if isinstance(handle, plt.Line2D):  # Check if it's a Line2D object
        handle.set_linewidth(LINEWIDTH+1)  # Set the desired linewidth

wrapped_labels = [fill(label, width=25) for label in labels]  # Set `width` to control wrap length

# Add the legend from the previous figure to this figure
ax4.legend(handles, labels, ncol=2, fontsize=FONTSIZE+2, loc='upper center', bbox_to_anchor=(0.5, 1),
           handletextpad=0.8,          # Adjust space between legend handles and text
           columnspacing=0.6,          # Adjust spacing between columns
           borderaxespad=0.0           # Minimize space between legend and axes
           )
# Turn off the axes
ax4.axis('off')

fig4.savefig(cwd / "plots" / "legend.svg", bbox_inches="tight", format="svg")
