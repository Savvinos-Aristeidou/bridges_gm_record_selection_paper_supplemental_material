import numpy as np
import pandas as pd



def get_hazard_curve_info(hazard_curve_path):
    hazard_curve_oq = pd.read_csv(hazard_curve_path)
    df = hazard_curve_oq.iloc[1:]
    f = hazard_curve_oq.keys()[-1]
    list_of_variables = f.split(',')
    investigation_time = float(list_of_variables[4].strip(' investigation_time='))
    imt = f.split('imt=')[-1][1:-1]
    # Remove 'poe-' from all elements
    IM = np.array([float(s.replace('poe-', '')) for s in hazard_curve_oq.iloc[0].tolist()[3:]])
    poes = np.array(hazard_curve_oq.iloc[1][3:]).astype(float)
    mafes = -np.log(1-poes)/investigation_time

    mask = ~((mafes == np.inf) | (mafes == 0))

    IM = IM[mask]
    mafes = mafes[mask]

    return IM, mafes
