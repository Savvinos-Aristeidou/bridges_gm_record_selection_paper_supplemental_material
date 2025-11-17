"""
author : JAWAD FAYAZ (email: jfayaz@uci.edu) (website: https://jfayaz.github.io) 
modified by: Savvinos Aristeidou

------------------------------ Description ---------------------------------------------
 
This is an associated file to generation of RotD(50 and 100) Spectra of bi-directional GM
It takes the path of an NGA-west2 acceleration time-history file
Outputs:    - Sampling interval [sec]
            - Number of points
            - Array of acceleration time-history points
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""
import numpy as np


def ReadGMFile(inFile='RSN1_HELENA.A_A-HMC180.at2'):
    with open(inFile, "r") as myfile:
        data = myfile.read().splitlines()
        sp = data[3].split()
        num_pts = int(sp[1].split(',')[0])
        dt = float(sp[3].split('SEC')[0])
        hdlines = 4

        for k in range(0, hdlines):
            del data[0]

        data = list(filter(str.strip, data))
        gm = np.array(' '.join(data).split()).astype(float)
        gmXY = gm

        del data
        del gm

        return dt, num_pts, gmXY
