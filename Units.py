"""
"""

import math

# --------------
# Basic Units
m = 1.0
kN = 1.0
sec = 1.0
LunitTXT = 'm'
FunitTXT = 'kN'
TunitTXT = 'sec'

# --------------
# Constants
pi = math.acos(-1)
g = 9.81*m/math.pow(sec, 2)
Ubig = 1e10
Usmall = 1/Ubig

# Length
mm = m/1000.0
cm = m/100.0
inch = 25.4*mm
ft = 12.0*inch

# Area
m2 = math.pow(m,2)
cm2 = math.pow(cm,2)
mm2 = math.pow(mm,2)
inch2 = math.pow(inch,2)

# First Moment of Area
m3 = math.pow(m,3)
cm3 = math.pow(cm,3)
mm3 = math.pow(mm,3)
inch3 = math.pow(inch,3)

# Second Moment of Area
m4 = math.pow(m,4)
cm4 = math.pow(cm,4)
mm4 = math.pow(mm,4)
inch4 = math.pow(inch,4)

# Force
N = kN/1000.0
kip = kN*4.448221615

# # Mass (tonnes)
tonne = 1.0
kg = tonne/1000

# Moment
kNm = kN*m

# Stress (kN/m2 or kPa)
Pa = N/(m2)
kPa = Pa*1.0e3
MPa = Pa*1.0e6
GPa = Pa*1.0e9
ksi = 6.8947573*MPa
psi = 1e-3*ksi

# Angles
deg = pi/180.0