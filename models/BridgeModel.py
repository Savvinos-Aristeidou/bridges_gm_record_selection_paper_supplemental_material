import os
import numpy as np
from openseespy import opensees as ops
from openseespy.postprocessing import Get_Rendering
from models.PierSection import PierSections
from Units import MPa, GPa, m, g, mm, pi
from analysis import Analysis

# This is a model of a bridge with a rectangular hollow section and variables
# pier heights
# ===========================================================================
# Written: Gerard O'Reilly
# IUSS Pavia
# November 2018
# Units in kN, m

# Define global variables and constants
# TODO: check the variables below
outsdir = "output_directory"  # TODO: specify your output directory
procdir = "proc_directory"    # TODO: specify your proc directory
# gmdir = "gm_directory"        # TODO: specify your GM directory
AType = "MSA"                 # specify Analysis Type
# model_num = 1                 # TODO: specify model number
# PierHeights = [1, 2, 3]       # TODO: example pier heights
# PierHeights = [2, 3, 3, 1, 3, 1, 2]       # TODO: example pier heights
ElementFormulation = "LumpedPlasticityModel"  # TODO: check if this is ok
# ElementFormulation = "FibreModel"  # TODO: check if this is ok
Lspan = 50 * m                    # Length of span
Htyp = 7 * m                      # Height type
Ecd = 25 * GPa                # Young's modulus of the deck
Ad = 1.74e8 / Ecd                      # Cross-sectional area
Gc = 10 * GPa                      # Shear modulus
Jd = 1.17e8 / Gc                      # Torsional constant
Iyd = 2.21e9 / Ecd                     # Moment of inertia in Y direction
Izd = 1.34e8 / Ecd                     # Moment of inertia in Z direction
rhod = 17.4                    # Density of deck material
bp = 2 * m  # Section width
hp = 4 * m  # Section height
tp = 0.4 * m  # Wall thickness

Ap = hp*bp-(hp-2*tp)*(bp-2*tp)     # Pier cross sectional area
rhop = 2.4 * Ap                    # Mass per length of the pier
DeckMass = rhod * g * Lspan / g            # Mass of 1 single deck span
DeckMass_half = DeckMass / 2
integration1 = "Integration1"  # example integration method
integration2 = "Integration2"  # example integration method
integration3 = "Integration3"  # example integration method
g = 9.81                      # Gravity acceleration

# TODO: source Units.tcl
# TODO: source modalAnalysis.tcl


class Structure:

    def __init__(self, PierHeights, model_num):
        self.PierHeights = PierHeights
        self.model_num = model_num
        return

    def build_model(self):
        print(f"Analysing Bridge Model {self.model_num}")

        # Create output directory
        os.makedirs(outsdir, exist_ok=True)

        # Initialize the model
        ops.wipe()
        ops.model('basic', '-ndm', 3, '-ndf', 6)

        PierSections(ElementFormulation).define_sections()

        # --------------------------------------
        # Define geometry
        # --------------------------------------

        # TODO: Load the pier sections --> source PierSection.tcl
        Kpb = 26329  # Pot bearing stiffness

        nPs = len(self.PierHeights)

        # --------------------------------------
        # Define nodes and fixity
        # --------------------------------------
        # Define the beginning and end nodes
        # node  #   X               Y   Z
        ops.node(1, 0, 0, 0, '-mass', 0, 0, 0, 0, 0, 0)
        ops.node(2, 0, 0, 0, '-mass', 0, 0, 0, 0, 0, 0)
        ops.node(3, Lspan * (nPs + 1), 0, 0, '-mass', 0, 0, 0, 0, 0, 0)
        ops.node(4, Lspan * (nPs + 1), 0, 0, '-mass', 0, 0, 0, 0, 0, 0)
        # fix  node  dX  dY  dZ  rX  rY  rZ
        ops.fix(1, 1, 1, 1, 1, 1, 1)
        ops.fix(2, 0, 0, 1, 0, 0, 0)
        ops.fix(3, 1, 1, 1, 1, 1, 1)
        ops.fix(4, 0, 0, 1, 0, 0, 0)

        # Define pier nodes and fixities
        for pier_num in range(1, nPs + 1):
            ops.node(10 + pier_num, pier_num * Lspan, 0.0, 0.0, '-mass', 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0)  # Top pier node
            ops.node(20 + pier_num, pier_num * Lspan, 0.0, -Htyp *
                    self.PierHeights[pier_num - 1], '-mass', 0.0, 0.0,
                    0.0, 0.0, 0.0, 0.0)  # Bottome pier node
            ops.fix(20 + pier_num, 1, 1, 1, 1, 1, 1)  # Fix bottom node

        # --------------------------------------
        # Transformation Tags
        # --------------------------------------
        # geomTransf Linear $transfTag $vecxzX $vecxzY $vecxzZ <-jntOffset $dXi $dYi
        # $dZi $dXj $dYj $dZj>
        ops.geomTransf('Linear', 1, 0, 1, 0)
        ops.geomTransf('Corotational', 2, -1, 0, 0)

        # Define materials
        ops.uniaxialMaterial('Elastic', 200, Kpb)  # Kpb: Pot bearing stiffness

        # --------------------------------------
        # Define elements
        # --------------------------------------
        for pier_num in range(1, nPs + 1):
            kk = pier_num - 1
            jj = pier_num + 1
            # Create the deck
            if pier_num == 1:
                # Create the first piece of the deck from node 2 to the first
                # pier number
                ops.element('elasticBeamColumn', 10 + pier_num, 2, 10 + pier_num, Ad,
                            Ecd, Gc, Jd, Iyd, Izd, 1, '-mass', rhod, '-cMass')
            elif pier_num == nPs:
                # At the last pier, create the forward and backwards deck
                ops.element('elasticBeamColumn', 10 + pier_num, 10 + kk, 10 + pier_num,
                            Ad, Ecd, Gc, Jd, Iyd, Izd, 1, '-mass', rhod, '-cMass')
                ops.element('elasticBeamColumn', 10 + jj, 10 + pier_num, 4, Ad,
                            Ecd, Gc, Jd, Iyd, Izd, 1, '-mass', rhod, '-cMass')
            else:
                # Otherwise, just create the backwards one
                ops.element('elasticBeamColumn', 10 + pier_num, 10 + kk, 10 + pier_num,
                            Ad, Ecd, Gc, Jd, Iyd, Izd, 1, '-mass', rhod, '-cMass')
            # Create the piers
            pH = self.PierHeights[pier_num - 1]

            # TODO: change the pH options here
            if pH == 1:
                ops.element('forceBeamColumn', 20 + pier_num, 10 + pier_num, 20 + pier_num, 2, 1, '-mass', rhop, '-cMass')
            elif pH == 2:
                ops.element('forceBeamColumn', 20 + pier_num, 10 + pier_num, 20 + pier_num, 2, 2, '-mass', rhop, '-cMass')
            elif pH == 3:
                ops.element('forceBeamColumn', 20 + pier_num, 10 + pier_num, 20 + pier_num, 2, 3, '-mass', rhop, '-cMass')

        # Define pot bearings
        ops.element('zeroLength', 1, 1, 2, '-mat', 200, 200, '-dir', 1, 2)
        ops.element('zeroLength', 2, 3, 4, '-mat', 200, 200, '-dir', 1, 2)

        # --------------------------------------
        # Apply Masses
        # --------------------------------------
        # Commented out because Python equivalent of Tcl mass definition may vary
        # depending on requirements
        # TODO: Check how to apply the mass on the structure. And how to translate the
        # TODO: following code.
        # for {set pier_num 1} {$pier_num <= $nPs} {incr pier_num 1} {
        #   # Take 1/3 of the pier mass and the full deck contribution
        #   mass 1${pier_num}   0. [expr 0.333*$rhop*[lindex $PierHeights $pier_num-1]*
        #   $Htyp+$DeckMass] 0.  0. 0. 0.
        # }
        # mass 2 0. $DeckMass_half 0.   0. 0. 0.
        # mass 4 0. $DeckMass_half 0.   0. 0. 0.

        # --------------------------------------
        # Apply Gravity Loading
        # --------------------------------------
        ops.timeSeries('Linear', 1)
        ops.pattern('Plain', 1, 1)

        for pier_num in range(1, len(self.PierHeights) + 1):
            ops.load(10 + pier_num, 0.0, 0.0, -((rhop * self.PierHeights[pier_num - 1] *
                                                Htyp) + DeckMass) * g, 0.0, 0.0, 0.0)
        ops.load(2, 0.0, 0.0, -DeckMass_half * g, 0.0, 0.0, 0.0)
        ops.load(4, 0.0, 0.0, -DeckMass_half * g, 0.0, 0.0, 0.0)

        # Define analysis parameters
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('UmfPack')
        ops.test('NormDispIncr', 1e-6, 500)
        ops.algorithm('Newton')
        nG = 100
        ops.integrator('LoadControl', 1 / nG)
        ops.analysis('Static')
        ops.analyze(nG)
        ops.loadConst('-time', 0.0)

        # Print model
        with open(os.path.join(outsdir, f"BridgeModel_{self.model_num}.txt"), 'w') as f:
            f.write("Model printed successfully.\n")
            # or use a more detailed method to print the model if needed

    def modal_analysis(self, num_modes, filename):
        # --------------------------------------
        # Modal analysis
        # --------------------------------------
        # Assuming modalAnalysis is a custom function or needs to be implemented
        # TODO: Implement modal analysis function here
        # modal_analysis(9, 0, f"_BridgeModel_{self.model_num}_{run}", outsdir, {})
        omega_sq = np.array(ops.eigen('-genBandArpack', num_modes))
        omega = np.sqrt(omega_sq)
        T = 2 * np.pi / omega
        ops.modalProperties('-file', filename)
        return omega

    def analyse(self, GMdatabase_dir, EQnameX, omega, SF):
        # --------------------------------------
        # DEFINE TIME SERIES
        # --------------------------------------
        pTagX = 1  # Set a pattern tag in X
        pTagY = 2  # Set a pattern tag in Y
        tsTagX = 1  # Set a timeseries tag,
        # this is needed for total floor acceleration recorder
        tsTagY = 2  # Set a timeseries tag,
        # this is needed for total floor acceleration recorder

        # if AType in ["NRHA", "IDA", "MSA"]:
        #     # TODO: assing the dt value of each record, instead of hardcoding 0.01
        #     # TODO: use sfX and sfY, instead of 1
        #     ops.timeSeries('Path', tsTagX, '-dt', 0.01, '-filePath',
        #                 f"{gmdir}/EQnameX", '-factor', 1)
        #     ops.timeSeries('Path', tsTagY, '-dt', 0.01, '-filePath',
        #                 f"{gmdir}/EQnameY", '-factor', 1)
        # elif AType == "SPO":
        #     ops.timeSeries('Linear', tsTagY, '-factor', 1)

        # ops.timeSeries('Path', tsTagY, '-dt', dt, '-filePath', f"{gmdir}/EQnameY", '-factor', SF*g)
        # --------------------------------------
        # DEFINE RECORDERS
        # --------------------------------------
        deck_node = [2] + [10 + i for i in range(1, len(self.PierHeights) + 1)] + [4]
        base_node = [1] + [20 + i for i in range(1, len(self.PierHeights) + 1)] + [3]
        pier_ele = [20 + i for i in range(1, len(self.PierHeights) + 1)]
        deck_ele = [10 + i for i in range(1, len(self.PierHeights) + 1)]
        tNode = [10 + i for i in range(1, len(self.PierHeights) + 1)]
        bNode = [20 + i for i in range(1, len(self.PierHeights) + 1)]

        analyse = Analysis()
        # TODO: run runNRHA3D
        # TODO: maybe damping should be changed to 2%
        curvature = analyse.perform_NRHA(GMdatabase_dir, EQnameX, 0.05, omega, SF, 10, tNode, bNode, pier_ele)

        return curvature

    def pushover(self):
        print("Need to fix up the static pushover parameters...")
        # Implement Static Pushover Parameters here
        # TODO: check what to do here, because I don't really get it
        # # Define the lateral load pattern
        # def define_lateral_load_pattern(pattern_name, pTag, tsTag, deck_node, deck_wt):
        #     # This function defines a lateral load pattern based on given nodes and weights
        #     for deck_pt in range(len(deck_node)):
        #         node = deck_node[deck_pt]
        #         weight = deck_wt[deck_pt]
        #         # Assuming `load` is a function or method that applies loads to the model
        #         load(node, 0.0, weight, 0.0, 0.0, 0.0, 0.0)

        # # Define the pattern
        # define_lateral_load_pattern('Plain', pTag, tsTag, deck_node, deck_wt)

        # # Push-over analysis setup
        # def setup_pushover_analysis(procdir, tDisp, roofNode, nSteps):
        #     # Setting up the analysis configuration
        #     set_constraints('Transformation')
        #     set_numberer('RCM')
        #     set_system('UmfPack')

        #     # Load and execute the push-over analysis script
        #     exec(open(f"{procdir}/singlePush.py").read())
        #     singlePush(0.001, tDisp, roofNode, 1, nSteps, 1)

        # # Setup pushover analysis
        # setup_pushover_analysis(procdir, tDisp, roofNode, nSteps)
