"""
Library of molecules from MolSSI QuESt
"""

water_molecule = """
O
H 1 1.1
H 1 1.1 2 104
"""

co_molecule = """
C
O 1 0.955
"""

co2_molecule = """
C
O 1 1.16
O 1 1.16 2 180
"""

acetylene_molecule = """
H
C 1 HC
X 2 XL 1 A1
C 2 CC 3 A1 1 D1
X 3 XL 2 A1 4 D2
H 4 HC 5 A1 2 D1

HC = 1.08
XL = 1.0
CC = 1.2
A1 = 90.0
D1 = 180.0
D2 = 0.0
"""

benzene_molecule = """
X
X 1 1.0
C 2 XC 1 A1
C 2 XC 1 A1 3 60.0
C 2 XC 1 A1 4 60.0
C 2 XC 1 A1 5 60.0
C 2 XC 1 A1 6 60.0
C 2 XC 1 A1 7 60.0
X 3 1.0 2 A1 1 0.0
H 3 HC 9 A1 2 180.0
H 4 HC 3 A2 2 180.0
H 5 HC 4 A2 2 180.0
H 6 HC 5 A2 2 180.0
H 7 HC 6 A2 2 180.0
H 8 HC 7 A2 2 180.0

A1 = 90.0
A2 = 120.0
XC = 1.3
HC = 1.08

"""

ethylene_molecule = """
H
C 1 HC
H 2 HC 1 A1
C 2 CC 3 A1 1 D1
H 4 HC 2 A1 1 D1
H 4 HC 2 A1 1 D2

HC = 1.08
CC = 1.4
A1 = 120.0
D1 = 180.0
D2 = 0.0
"""

mollib = {}
mollib["H2O"] = water_molecule
#mollib["CO"] = co_molecule
mollib["CO2"] = co2_molecule
#mollib["C2H2"] = acetylene_molecule
mollib["C2H4"] = ethylene_molecule
#mollib["C6H6"] = benzene_molecule 

### NOTES ###
# CO
# # SCF/DZ could not converge to 1e-8, stuck in well (implement damping)
# C2H2
# # SCF/STO-3G could not converge to 1e-8 rms, 100 iterations
# C6H6
# # SCF could not converge to 1e-6, 100 iterations
