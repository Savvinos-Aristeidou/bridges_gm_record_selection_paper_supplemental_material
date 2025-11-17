import json
import os
import pickle
import time
from matplotlib import pyplot as plt
import numpy as np
from openseespy import opensees as ops
from pathlib import Path
from analysis import Analysis
from models.BridgeModel import Structure
from models.PierSection import PierSections, fc as pier_fc, bp as pier_bp, hp as pier_hp, tp as pier_tp, Ap as pier_Ap, rhop as pier_rhop, cv as pier_cv, db as pier_db
from Units import MPa, GPa, m, g, mm, pi
import math

# Master file to run analysis on bridges
# Written: Gerard O'Reilly
# IUSS Pavia
# November 2018
# Units in kN, m

print("""
    # --------------------------------------------------
    # Master file to run analysis on bridges
    # --------------------------------------------------
    # Gerard O'Reilly
    # IUSS Pavia
    # --------------------------------------------------
    """)

cwd = Path.cwd()  # Include the used procedures in the current working directory

# --------------------------------------
# Static Pushover Analysis of Single Piers
# --------------------------------------
def static_pushover_analysis():
    # Pier Pushover Analysis
    outsdir = "outs_PierPushover"
    os.makedirs(outsdir, exist_ok=True)

    force = []
    displacement = []
    for d in range(2, 3):
        for pH in range(1, 2):
            # for ElementFormulation in ["FibreModel", "LumpedPlasticityModel"]:
            for ElementFormulation in ["LumpedPlasticityModel"]:

                Lspan = 50 * m                    # Length of span
                Htyp = 7 * m                      # Height type
                rhod = 17.4                    # Density of deck material
                bp = 2 * m  # Section width
                hp = 4 * m  # Section height
                tp = 0.4 * m  # Wall thickness

                cv = 20.0 * mm  # Concrete cover
                db = 20.0 * mm  # Bar diameter
                fc = 42.0 * MPa  # Unconfined compressive strength
                k_conf = 1.2  # Confinement factor
                fc_conf = k_conf * fc


                Ap = hp*bp-(hp-2*tp)*(bp-2*tp)     # Pier cross sectional area
                rhop = 2.4 * Ap                    # Mass per length of the pier

                mpier = rhod * Lspan  # Replace with actual rho and Lspan if needed

                Paxial = - mpier * g  # Replace 1 with actual m if needed, also the g value

                # rhod and rhop are defined in BridgeModel.py
                fy = 500.0 * MPa  # Yield strength
                Es = 200.0 * GPa  # Elastic modulus

                pier_sections = PierSections(ElementFormulation)
                # Model setup

                ops.wipe()
                ops.model('basic', '-ndm', 3, '-ndf', 6)
                ops.node(1, 0, 0, 0)
                ops.node(2, 0, 0, Htyp * pH)
                ops.fix(1, 1, 1, 1, 1, 1, 1)
                ops.fix(2, 0, 0, 0, 0, 0, 0)

                pier_sections.define_sections()

                ops.geomTransf('Corotational', 2, -1, 0, 0)
                if pH == 1:
                    ops.element('forceBeamColumn', 1, 2, 1, 2, 1, '-mass', rhop, '-cMass')
                elif pH == 2:
                    ops.element('forceBeamColumn', 1, 2, 1, 2, 2, '-mass', rhop, '-cMass')
                elif pH == 3:
                    ops.element('forceBeamColumn', 1, 2, 1, 2, 3, '-mass', rhop, '-cMass')


                # Set loading pattern
                # Placeholder for loading pattern and constraints
                # Example: apply_load(Paxial)
                # ops.wipeAnalysis()
                ops.timeSeries('Linear', 1)
                ops.pattern('Plain', 1, 1)
                ops.load(2, 0, 0, Paxial, 0, 0, 0)

                ops.constraints('Transformation')
                ops.numberer('RCM')
                ops.system('FullGeneral')
                ops.test('NormDispIncr', 1e-6, 500)
                ops.algorithm('Newton')

                # Analysis settings
                nG = 100

                # Placeholder for running analysis
                # Example: analyze_model(nG)

                ops.integrator('LoadControl', 1/nG)
                ops.analysis('Static')
                ops.analyze(nG)
                ops.loadConst('-time', 0)
                # Reset or clear model state
                # Example: reset_model()
                # Define reference moment

                ops.timeSeries('Linear', 2)
                ops.pattern('Plain', 2, 2)
                if d == 1:
                    ops.load(2, 1, 0, 0, 0, 0, 0)
                elif d == 2:
                    ops.load(2, 0, 1, 0, 0, 0, 0)
                # ops.constraints('Plain')
                # ops.numberer('Plain')
                # ops.system('UmfPack')

                # Placeholder for singlePush analysis
                phiYzz = 2.1 * fy / Es / hp  # Estimate the yield curvature of the section
                phiYyy = 2.1 * fy / Es / bp  # Estimate the yield curvature of the section
                # fy, Es, and hp defined in PierSection.py
                # singlePush dref mu ctrlNode dispDir nSteps {IOflag 1} {PrintFlag 0}
                analyse = Analysis()
                if d == 1:
                    # Placeholder for singlePush analysis. Import the function
                    # Example: single_push_analysis(phiYzz, mu)
                    # singlePush(phiYzz, mu, 2, 6, 200, 1)
                    LoadFactor, DispCtrlNode = analyse.perform_cyclic_push(dref=0.2*m, mu=1, numCycles=0, ctrlNode=2, dispDir=1, dispIncr=20000, IOflag=1, PrintFlag=1)
                elif d == 2:
                    # Placeholder for singlePush analysis
                    # Example: single_push_analysis(phiYyy, mu)
                    # singlePush(phiYyy, mu, 2, 5, 200, 1)
                    LoadFactor, DispCtrlNode = analyse.perform_cyclic_push(dref=0.2*m, mu=1, numCycles=0, ctrlNode=2, dispDir=2, dispIncr=20000, IOflag=1, PrintFlag=1)

                force.append(LoadFactor)
                displacement.append(DispCtrlNode)
                ops.wipe()

        # Compute shear capacity according to Eq. (11-4) (members subject to axial compression)
        def compute_shear_ACI_318_11(fc_val, Nu_kN, Ag, b_w, d_eff, lambda_factor=1.0, phi=0.75):
            """Compute Vc from ASCE/ACI Eq.11-4 and return phi*Vc and Vc in kN.

            fc_val : concrete strength value taken from model (project units)
            Nu_kN  : axial force (positive in compression) in kN
            Ag     : gross area (m^2)
            b_w    : web width (m) (use wall thickness for hollow section)
            d_eff  : effective depth (m)
            lambda_factor: lightweight concrete factor (1.0 for normal-weight)
            phi    : strength reduction factor (default 0.75)
            """
            # The code's Eq.11-4 uses sqrt(f'c) with f'c in psi and b_w,d in inches.
            # fc_val in this project is in Pa (N/m^2) because PierSection defines fc = 42.0*MPa.
            # Steps:
            # 1) Convert fc (Pa) -> psi
            # 2) Convert b_w and d_eff from m -> inches
            # 3) Compute Vc in lb using ACI form: Vc_lb = 2 * axial_factor * lambda * sqrt(fc_psi) * b_w_in * d_in
            # 4) Convert Vc_lb to kN for plotting (1 lb = 4.448221615 N)

            # Convert fc to Pa then to psi
            # NOTE: in this codebase fc_val is provided in kN/m^2 (project units),
            # so convert to Pa by multiplying by 1000.
            fc_Pa = float(fc_val) * 1000.0
            fc_psi = fc_Pa / 6894.75729

            # Use axial force computed earlier (Paxial) — take absolute (compression positive)
            # Convert axial Nu from kN to N
            Nu_N = abs(Nu_kN) * 1000.0

            # Nu/Ag in Pa -> convert to psi
            Nu_over_Ag_Pa = Nu_N / float(Ag)
            Nu_over_Ag_psi = Nu_over_Ag_Pa / 6894.75729

            # axial factor (1 + Nu/(2000 Ag)) where Nu/Ag is in psi
            axial_factor = 1.0 + (Nu_over_Ag_psi / 2000.0)

            # Convert b_w and d_eff from meters to inches (1 m = 39.37007874 in)
            m_to_in = 39.37007874
            b_w_in = b_w * m_to_in
            d_in = d_eff * m_to_in

            # Compute Vc in lb using ACI expression
            Vc_lb = 2.0 * axial_factor * lambda_factor * np.sqrt(fc_psi) * (b_w_in) * (d_in)

            # Convert lb to N then to kN
            Vc_N = Vc_lb * 4.448221615
            Vc_kN = Vc_N / 1000.0

            Vphi_kN = phi * Vc_kN

            # Eq (11-13):
            Av_min = 0.75*np.sqrt(fc_psi)*b_w_in*6/(500*10**6/6894.75729)

            # Selected nominal area: we select bar number #3 and it have 4 legs, therefore 0.11 in2 * 4 transverse reinforcement
            Vs = 0.11*4*(500*10**6/6894.75729)*d_in/6
            Vs_N = Vs * 4.448221615
            Vs_kN = Vs_N / 1000.0

            V_strength_kN = Vc_kN + Vs_kN

            return V_strength_kN

        # Effective depth approximation: d = h/2 - cover - 0.5*db

        d_eff = max(1e-6, hp - cv - 0.5 * db)

        V_strength_kN = compute_shear_ACI_318_11(fc_conf, -((rhop * 1 * Htyp * m) + mpier) * g, Ap, tp*2, d_eff, lambda_factor=1.0, phi=0.75)

        # Plot pushover curve and shear capacity
        plt.figure(figsize=(3.9, 3.0))
        plt.plot(displacement[0], -force[0], label='Pushover curve', linewidth=1.5)
        plt.axhline(V_strength_kN, color='r', linestyle='--', linewidth=1.5, label=f"$V_R$ (Eqs.11-4 & 11-15) = {V_strength_kN:.1f} kN")
        plt.grid()
        plt.legend(fontsize=9)
        plt.xlabel('Control displacement [m]', fontsize=9)
        plt.ylabel('Base shear [kN]', fontsize=9)
        plt.title('Pushover curve with design shear capacity', fontsize=9)
        plt.tick_params(axis='both', which='both', labelsize=9)
        plt.savefig(cwd / "plots" / "pushover_with_shear_capacity.svg", bbox_inches="tight", format="svg")

        exit()


# --------------------------------------
# Modal Analysis
# --------------------------------------
def modal_analysis():
    AType = "ModalAnalysis"
    outsdir = "outs_ModalAnalysis2"
    ElementFormulation = "LumpedPlasticityModel"
    run = 1
    # ElementFormulation = "FibreModel"
    # run = 2

    for model_num in range(1, 8):
        if model_num == 1:
            PierHeights = [1, 2, 3]
        elif model_num == 2:
            PierHeights = [2, 1, 3]
        elif model_num == 3:
            PierHeights = [2, 2, 2]
        elif model_num == 4:
            PierHeights = [2, 3, 2]
        elif model_num == 5:
            PierHeights = [2, 2, 2, 2, 2, 2, 2]
        elif model_num == 6:
            PierHeights = [2, 3, 3, 1, 3, 1, 2]
        elif model_num == 7:
            PierHeights = [3, 3, 3, 2, 1, 1, 1]

        # TODO: Source BridgeModel.tcl equivalent function
        ops.wipe()

# --------------------------------------
# IDA with selected ground motions
# --------------------------------------
def IDA_HTF(firstInt, incrStep, maxRuns, IMtype, Tinfo, xi, Dt, dCap, gmsdir,
            nmsfileX, nmsfileY, dtsfile, dursfile, outsdir, mdlfile, pflag=0):
    # Placeholder function for IDA_HTF
    # TODO: call this function, belonging to another file
    pass


def IDA_analysis():
    AType = "IDA"
    # TODO: source runNRHA3D.tcl
    # TODO: source getSaT.tcl
    # TODO: source IDA_HTF.tcl

    IMlist = ["PGA", "PGV", "SaT1", "SaTmed", "AvgSa-single", "AvgSa-group"]
    # IMlist = ["AvgSa-group"]

    for IMcurr in IMlist:
        for model_num in range(1, 8):
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f"Performing IDA analysis on Bridge:{model_num} with {IMcurr}")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")

            if model_num == 1:
                PierHeights = [1, 2, 3]
            elif model_num == 2:
                PierHeights = [2, 1, 3]
            elif model_num == 3:
                PierHeights = [2, 2, 2]
            elif model_num == 4:
                PierHeights = [2, 3, 2]
            elif model_num == 5:
                PierHeights = [2, 2, 2, 2, 2, 2, 2]
            elif model_num == 6:
                PierHeights = [2, 3, 3, 1, 3, 1, 2]
            elif model_num == 7:
                PierHeights = [3, 3, 3, 2, 1, 1, 1]

            # Set up the outputs and source the model file
            outsdir = f"outs_{AType}_Bridge{model_num}_{IMcurr}"
            ElementFormulation = "LumpedPlasticityModel"
            mdlfile = "models/BridgeModel.tcl"

            # Get the ground motions to be used
            if IMcurr == "PGA":
                print("Using PGA")
                IMtype = 1
                Tinfo = []
                # TODO: assign the correct directories here for ground motions
                gmsdir = "../OQ-GMs/PGA-site_0-poe-0.02"
            elif IMcurr == "PGV":
                print("Using PGV")
                IMtype = 4
                Tinfo = []
                gmsdir = "../OQ-GMs/PGA-site_0-poe-0.02"
            elif IMcurr == "SaTmed":
                print("Using SaTmed")
                IMtype = 2
                gmsdir = "../OQ-GMs/SA(0.47)-site_0-poe-0.02"
                Tinfo = [0.47]
            elif IMcurr == "SaT1":
                print("Using SaT1")
                IMtype = 2
                gmsdir, Tinfo = get_gmsdir_and_tinfo_for_SaT1(model_num)
            elif IMcurr == "AvgSa-single":
                print("Using AvgSa-single")
                IMtype = 3
                gmsdir, Tinfo = get_gmsdir_and_tinfo_for_AvgSa_single(model_num)
            elif IMcurr == "AvgSa-group":
                print("Using AvgSa-group")
                IMtype = 3
                gmsdir = "../OQ-GMs/AvgSA-119-site_0-poe-0.02"
                Tinfo = [0.11, 0.19, 0.27, 0.35, 0.43, 0.51, 0.59, 0.67, 0.75, 0.83]

            # Set the HTF parameters
            if IMcurr == "PGV":
                firstInt = 0.1
                incrStep = 0.5
            else:
                firstInt = 0.05
                incrStep = 0.25

            # TODO: same ground motions in X and Y?
            nmsfile = "GMR_names1.txt"
            dtsfile = "GMR_dts.txt"
            dursfile = "GMR_durs.txt"
            # IDA_HTF firstInt  incrStep  maxRuns   IMtype Tinfo   xi  Dt  dCap gmsdir nmsfileX nmsfileY dtsfile dursfile outsdir mdlfile {pflag 0}
            IDA_HTF(firstInt, incrStep, 15, IMtype, Tinfo, 0.05, 0.01, 10.0, gmsdir, nmsfile, nmsfile, dtsfile, dursfile, outsdir, mdlfile, 1)
            ops.wipe()


def get_gmsdir_and_tinfo_for_SaT1(model_num):
    if model_num in {1, 2, 7}:
        return "../OQ-GMs/SA(0.56)-site_0-poe-0.02", [0.56]
    elif model_num == 3:
        return "../OQ-GMs/SA(0.48)-site_0-poe-0.02", [0.48]
    elif model_num == 4:
        return "../OQ-GMs/SA(0.51)-site_0-poe-0.02", [0.51]
    elif model_num in {5, 6}:
        return "../OQ-GMs/SA(0.49)-site_0-poe-0.02", [0.49]


def get_gmsdir_and_tinfo_for_AvgSa_single(model_num):
    if model_num == 1:
        return ("../OQ-GMs/AvgSA-92-site_0-poe-0.02",
                [0.14, 0.22, 0.29, 0.37, 0.45, 0.52, 0.60, 0.68, 0.76, 0.83])
    elif model_num == 2:
        return ("../OQ-GMs/AvgSA-93-site_0-poe-0.02",
                [0.13, 0.20, 0.28, 0.36, 0.44, 0.52, 0.60, 0.68, 0.75, 0.83])
    elif model_num == 3:
        return ("../OQ-GMs/AvgSA-94-site_0-poe-0.02",
                [0.11, 0.18, 0.25, 0.32, 0.38, 0.45, 0.52, 0.59, 0.66, 0.72])
    elif model_num == 4:
        return ("../OQ-GMs/AvgSA-95-site_0-poe-0.02",
                [0.15, 0.22, 0.29, 0.36, 0.42, 0.49, 0.56, 0.63, 0.69, 0.76])
    elif model_num == 5:
        return ("../OQ-GMs/AvgSA-96-site_0-poe-0.02",
                [0.11, 0.18, 0.25, 0.31, 0.38, 0.45, 0.52, 0.58, 0.65, 0.72])
    elif model_num == 6:
        return ("../OQ-GMs/AvgSA-97-site_0-poe-0.02",
                [0.18, 0.24, 0.30, 0.37, 0.43, 0.49, 0.55, 0.62, 0.68, 0.74])
    elif model_num == 7:
        return ("../OQ-GMs/AvgSA-98-site_0-poe-0.02",
                [0.19, 0.26, 0.34, 0.41, 0.48, 0.55, 0.62, 0.69, 0.76, 0.83])


# print(" # -------------------------------------------------- ")
# print(" # Analysis complete                                 ")
# print(" # -------------------------------------------------- ")

# --------------------------------------
# MSA with selected ground motions
# --------------------------------------
def run_MSA():
    AType = "MSA"
    # TODO: source runNRHA3D.tcl procedure

    # IMlist = ["PGA", "PGV", "SaT1", "SaTmed", "AvgSa-single", "AvgSa-group"]
    # IMlist = ["PGA", "PGV"]
    # IMlist = ["SaT1", "SaTmed", "AvgSa-single", "AvgSa-group"]
    # IMlist = ["SaT1"]
    IMlist = ["input0"]
    # IMlist = ["Sa01s", "Sa02s", "Sa03s"]
    # IMlist = ["Sa04s", "Sa05s", "Sa06s"]
    # IMlist = ["Sa07s", "Sa08s", "Sa09s"]
    # IMlist = ["Sa10s", "Sa12s", "Sa15s", "Sa20s", "Sa25s"]
    bridge_num = 7

    poes = [0.9, 0.7, 0.5, 0.3, 0.2, 0.1, 0.05, 0.02, 0.01]  # investigation time 50 yrs
    # poes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.0025, 0.001]
    # poes = [0.0025, 0.001]
    # poes [0.5]
    # TODO: do not hardcode the number of ground motions
    nGMRs = 50
    count = 0

    GMdatabase_dir = cwd.parent / "NGA_West2_database"
    metadata_file_path = cwd.parent / "Database_writing/NGA_W2_7.pickle"
    with open(metadata_file_path, "rb") as f:
        metadata = pickle.load(f)

    for IMcurr in IMlist:
        for model_num in range(bridge_num, bridge_num+1):
            if model_num == 1:
                PierHeights = [1, 2, 3]
            elif model_num == 2:
                PierHeights = [2, 1, 3]
            elif model_num == 3:
                PierHeights = [2, 2, 2]
            elif model_num == 4:
                PierHeights = [2, 3, 2]
            elif model_num == 5:
                PierHeights = [2, 2, 2, 2, 2, 2, 2]
            elif model_num == 6:
                PierHeights = [2, 3, 3, 1, 3, 1, 2]
            elif model_num == 7:
                PierHeights = [3, 3, 3, 2, 1, 1, 1]

            outsdir = f"outs_{AType}_Bridge{model_num}_{IMcurr}"
            os.makedirs(outsdir, exist_ok=True)
            # ElementFormulation = "LumpedPlasticityModel"
            # mdlfile = "models/BridgeModel.tcl"

            # Loop through each IM level
            for IML, poe_curr in enumerate(poes, start=1):

                if IMcurr == "Sa1s-Disagg-Select":
                    gmsdir = f"../OQ-GMs/SA(1.0)-{poe_curr}-Disagg-Select/GMRs"

                GM_input_info_path = cwd / "GM_input" / f"Bridge{model_num}" / f"rec_{IMcurr}_IML{IML}.json"

                with open(GM_input_info_path) as f:
                    GM_input_info = json.load(f)

                # TODO: only input the x earthquake for now
                # Set the GMR file parameters
                EQnameX_list = GM_input_info['selected_scaled_best']['filenames']
                SFs_list = GM_input_info['selected_scaled_best']['SF_opt']

                # dts_list = open(f"{gmsdir}/dt.txt").read().splitlines()
                # durs_list = open(f"{gmsdir}/dur.txt").read().splitlines()

                # Open logfile
                logfile = f"{outsdir}/IM_POE_{poe_curr}.txt"
                curvature = np.full(nGMRs, np.nan)
                # Loop through each ground motion
                for q in range(1, nGMRs + 1):
                    # Print the progress
                    current_progress = 100 * count / (len(IMlist) * 7 *
                                                      len(poes) * nGMRs)
                    # 7 is the number of models
                    print(str(f"Current progress: {current_progress:.2f}% - "
                              f"(IM:{IMlist.index(IMcurr)+1}/{len(IMlist)}  "
                              f"Model:{model_num}/7  PoE:{poes.index(poe_curr)+1}/"
                              f"{len(poes)}  GMR:{q}/{nGMRs})"))

                    count += 1

                    # TODO: Only eq in x direction for now
                    # Get the current record
                    EQnameX = EQnameX_list[q-1]  # Get the name of the record
                    EQnameY = EQnameX_list[q-1]  # Get the name of the record
                    # dt = dts_list[q-1]  # Current dt
                    # dur = durs_list[q-1]  # Current duration
                    # sfX = 9.81  # Scale factor
                    # sfY = 9.81  # Scale factor
                    # Tmax = float(dur) + 0.0

                    # Source the model and run the analysis

                    # Here, you would call the model and analysis functions
                    # Placeholder for sourcing model file and running analysis
                    # Example: source_and_run_analysis(mdlfile, dt, Tmax, logfile, etc.)

                    bridge = Structure(PierHeights, model_num)
                    bridge.build_model()
                    omega = bridge.modal_analysis(9, f'ModalReport{model_num}.txt')
                    curvature[q-1] = bridge.analyse(GMdatabase_dir, EQnameY, omega, SFs_list[q-1])

                # Write the 1D array to the file
                with open(logfile, "w") as f:
                    for item in curvature:
                        f.write(f"{item}\n")
            # if model_num == 4:
            #     exit()

    print("Current progress: 100% - DONE")


def run_moment_curvature_analysis():
    # Moment Curvature Analysis
    outsdir = "outs_MomentCurvature"
    os.makedirs(outsdir, exist_ok=True)
    mu = 50.0

    # TODO: source Units.tcl
    # TODO: source singlePush.tcl
    # Replace sourcing of Tcl files with equivalent Python code if available
    # Example: import Units.tcl and singlePush.tcl
    moment=[]
    curvature=[]
    for d in range(1, 3):
        for pH in range(1, 2):
            for ElementFormulation in ["FibreModel", "LumpedPlasticityModel"]:
            # for ElementFormulation in ["FibreModel"]:
                pier_sections = PierSections(ElementFormulation)
                # Model setup
                # Placeholder for model setup
                # Replace with actual calls to create nodes, elements, etc.
                # Example: model = create_model()
                ops.wipe()
                ops.model('basic', '-ndm', 3, '-ndf', 6)
                ops.node(1, 0, 0, 0)
                ops.node(2, 0, 0, 0)
                ops.fix(1, 1, 1, 1, 1, 1, 1)
                if d == 1:
                    ops.fix(2, 0, 1, 1, 1, 1, 0)
                elif d == 2:
                    ops.fix(2, 0, 1, 1, 1, 0, 1)
                # TODO: source PierSection.tcl
                pier_sections.define_sections()
                if ElementFormulation == "FibreModel":
                    ops.element('zeroLengthSection', 1, 1, 2, 1)
                elif ElementFormulation == "LumpedPlasticityModel":
                    ops.element('zeroLengthSection', 1, 1, 2, int(f"30{pH}"))

                # ops.recorder('Node', '-file', outsdir + f"/F_{pH}_{d}.txt",
                #             '-node', 1, '-dof', 1, 2, 3, 4, 5, 6, 'reaction')
                # ops.recorder('Node', '-file', outsdir + f"/D_{pH}_{d}.txt",
                #             '-node', 2, '-dof', 1, 2, 3, 4, 5, 6, 'disp')

                Lspan = 50 * m                    # Length of span
                Htyp = 7 * m                      # Height type
                rhod = 17.4                    # Density of deck material
                bp = 2 * m  # Section width
                hp = 4 * m  # Section height
                tp = 0.4 * m  # Wall thickness

                Ap = hp*bp-(hp-2*tp)*(bp-2*tp)     # Pier cross sectional area
                rhop = 2.4 * Ap                    # Mass per length of the pier

                mpier = rhod * Lspan  # Replace with actual rho and Lspan if needed

                Paxial = -((rhop * pH * Htyp * m) + mpier) * g  # Replace 1 with actual m if needed, also the g value

                # rhod and rhop are defined in BridgeModel.py
                fy = 500.0 * MPa  # Yield strength
                Es = 200.0 * GPa  # Elastic modulus

                # Set loading pattern
                # Placeholder for loading pattern and constraints
                # Example: apply_load(Paxial)
                # ops.wipeAnalysis()
                ops.timeSeries('Linear', 1)
                ops.pattern('Plain', 1, 1)
                ops.load(2, Paxial, 0, 0, 0, 0, 0)

                ops.constraints('Transformation')
                ops.numberer('RCM')
                ops.system('FullGeneral')
                ops.test('NormDispIncr', 1e-6, 500)
                ops.algorithm('Newton')

                # Analysis settings
                nG = 10

                # Placeholder for running analysis
                # Example: analyze_model(nG)

                ops.integrator('LoadControl', 1/nG)
                ops.analysis('Static')
                ops.analyze(nG)
                ops.loadConst('-time', 0)
                # Reset or clear model state
                # Example: reset_model()
                # Define reference moment

                ops.timeSeries('Linear', 2)
                ops.pattern('Plain', 2, 2)
                if d == 1:
                    ops.load(2, 0, 0, 0, 0, 0, 1)
                elif d == 2:
                    ops.load(2, 0, 0, 0, 0, 1, 0)
                # ops.constraints('Plain')
                # ops.numberer('Plain')
                # ops.system('UmfPack')

                # Placeholder for singlePush analysis
                phiYzz = 2.1 * fy / Es / hp  # Estimate the yield curvature of the section
                phiYyy = 2.1 * fy / Es / bp  # Estimate the yield curvature of the section
                # fy, Es, and hp defined in PierSection.py
                # singlePush dref mu ctrlNode dispDir nSteps {IOflag 1} {PrintFlag 0}
                analyse = Analysis()
                if d == 1:
                    # Placeholder for singlePush analysis. Import the function
                    # Example: single_push_analysis(phiYzz, mu)
                    # singlePush(phiYzz, mu, 2, 6, 200, 1)
                    LoadFactor, DispCtrlNode = analyse.perform_cyclic_push(dref=phiYzz, mu=mu, numCycles=6, ctrlNode=2, dispDir=6, dispIncr=200, IOflag=1, PrintFlag=1)
                elif d == 2:
                    # Placeholder for singlePush analysis
                    # Example: single_push_analysis(phiYyy, mu)
                    # singlePush(phiYyy, mu, 2, 5, 200, 1)
                    LoadFactor, DispCtrlNode = analyse.perform_cyclic_push(dref=phiYyy, mu=mu, numCycles=6, ctrlNode=2, dispDir=5, dispIncr=200, IOflag=1, PrintFlag=1)

                moment.append(LoadFactor)
                curvature.append(DispCtrlNode)
                ops.wipe()
            # fontsize = 9
            # linewidth = 1.5
            # plt.figure(figsize=(3.9, 3.0))
            # plt.plot(curvature[0]*1000, moment[0]/1000, linewidth=linewidth, label="Fibre section")
            # plt.plot(curvature[1]*1000, moment[1]/1000, linewidth=linewidth, label="Lumped\nplasticity hinge")
            # plt.xlim([-70, 70])
            # plt.ylabel("Moment, M [MNm]", fontsize=fontsize)
            # plt.xlabel("Curvature, φ [mrad]", fontsize=fontsize)
            # plt.legend(fontsize=fontsize, loc="upper left", frameon=False)
            # plt.grid()
            # plt.tick_params(axis='both', which='both', labelsize=fontsize)

            # plt.savefig(cwd / "plots" / "moment_curvature.svg", bbox_inches="tight", format="svg")

            # exit()
        fontsize = 9
        linewidth = 1.2
        plt.figure(figsize=(3.9, 3.0))
        # plt.plot(curvature[0]*1000, moment[0]/1000, linewidth=linewidth, label="Pier 1: h = 7 m")
        # plt.plot(curvature[1]*1000, moment[1]/1000, linewidth=linewidth, label="Pier 2: h = 14 m")
        # plt.plot(curvature[2]*1000, moment[2]/1000, linewidth=linewidth, label="Pier 3: h = 21 m")
        plt.plot(curvature[0]*1000, moment[0]/1000, linewidth=linewidth, label="Fibre section")
        plt.plot(curvature[1]*1000, moment[1]/1000, linewidth=linewidth, label="Lumped\nplasticity hinge")
        plt.xlim([-70, 70])
        plt.ylabel("Moment, M [MNm]", fontsize=fontsize)
        plt.xlabel("Curvature, φ [mrad]", fontsize=fontsize)
        plt.legend(fontsize=fontsize, loc="upper left", frameon=False)
        plt.grid()
        plt.tick_params(axis='both', which='both', labelsize=fontsize)

        plt.savefig(cwd / "plots" / "moment_curvature.svg", bbox_inches="tight", format="svg")

        exit()

# --------------------------------------
# Single NRHA
# --------------------------------------
def run_single_NRHA():
    # Single NRHA
    AType = "NRHA"
    outsdir = "outs_SingleRecord"
    os.makedirs(outsdir, exist_ok=True)
    # ElementFormulation = "LumpedPlasticityModel"
    # run = 1
    ElementFormulation = "FibreModel"
    run = 2

    for model_num in range(1, 8):
        PierHeights = {
            1: [1, 2, 3],
            2: [2, 1, 3],
            3: [2, 2, 2],
            4: [2, 3, 2],
            5: [2, 2, 2, 2, 2, 2, 2],
            6: [2, 3, 3, 1, 3, 1, 2],
            7: [3, 3, 3, 2, 1, 1, 1]
        }[model_num]

        # Load ground motion details
        # TODO: Change the directories here
        EQnameX = "/Users/Gerard/GitHub/Matlab-Functions/eq_1g_0.01s.txt"
        EQnameY = "/Users/Gerard/GitHub/Matlab-Functions/eq_1g_0.01s.txt"
        dt = 0.01  # Current dt
        Tmax = 70.0  # Current duration plus padding
        sfX = 0 * 9.81  # Scale factor
        sfY = 0.5 * 9.81  # Scale factor

        # TODO: Source the model file (not with exec command)
        # Example: source BridgeModel.tcl
        with open('BridgeModel.py') as file:
            exec(file.read())

        # TODO: source cwd/runNRHA3D.tcl
        # Placeholder for NRHA analysis

        log_filename = f"{outsdir}/logfile_{model_num}_{run}.txt"
        with open(log_filename, "w") as log:
            # Example: run_NRHA(log, dt, Tmax, tNode, bNode)
            runNRHA3D(0.01, Tmax, 20.0, tNode, bNode, log, 1)
        ops.wipe()

# --------------------------------------
# Incremental Dynamic Analysis
# --------------------------------------
def run_incremental_dynamic_analysis():
    # Incremental Dynamic Analysis
    AType = "IDA"
    # TODO: change the directories here
    gmdir = "/Users/Gerard/Desktop/FEMAP695"
    outsdir = "outs_IDA_FEMA_P695_Bridge"
    os.makedirs(outsdir, exist_ok=True)
    ElementFormulation = "LumpedPlasticityModel"

    # TODO: source the following:
    # source cwd/runNRHA3D.tcl
    # source cwd/getSaT.tcl
    # source cwd/IDA_HTF.tcl

    for model_num in range(1, 8):
        PierHeights = {
            1: [1, 2, 3],
            2: [2, 1, 3],
            3: [2, 2, 2],
            4: [2, 3, 2],
            5: [2, 2, 2, 2, 2, 2, 2],
            6: [2, 3, 3, 1, 3, 1, 2],
            7: [3, 3, 3, 2, 1, 1, 1]
        }[model_num]

        outsdir = "outs_IDA_FEMA_P695_Bridge${model_num}"

        # Load the information about the ground motions (FEMA P695 set)

        nmsfile = f"{gmdir}/FEMA_P695_unscaled_names.txt"
        dtsfile = f"{gmdir}/FEMA_P695_unscaled_dts.txt"
        dursfile = f"{gmdir}/FEMA_P695_unscaled_durs.txt"
        mdlfile = "BridgeModel.tcl"

        # Placeholder for IDA analysis
        # Example: run_IDA(0.05, 0.25, 15, 1, {}, 0.02, 0.01, 10.0, nmsfile, dtsfile, dursfile, outsdir, mdlfile, 1)
        # IDA_HTF firstInt incrStep maxRuns IMtype Tinfo   xi  Dt    dCap nmsfileX nmsfileY dtsfile dursfile outsdir mdlfile {pflag 0}
        IDA_HTF(0.05, 0.25, 15, 1, {}, 0.02, 0.01, 10.0, nmsfile, dtsfile, dursfile, outsdir, mdlfile, 1)
        ops.wipe()


if __name__ == "__main__":
    startTime = time.time()

    # Uncomment to run Static Pushover Analysis:
    static_pushover_analysis()

    # Uncomment to run Modal Analysis:
    # modal_analysis()

    # Uncomment to run IDA Analysis:
    # IDA_analysis()

    # Uncomment to run MSA Analysis:
    # run_MSA()

    # Uncomment for moment curvature analysis
    # run_moment_curvature_analysis()

    # Uncomment for single NRHA
    # run_single_NRHA()

    # Uncomment for IDA
    # run_incremental_dynamic_analysis()

    # --------------------------------------------------
    # Stop the clock
    # --------------------------------------------------
    endTime = time.time()
    totalTimeSEC = endTime - startTime
    totalTimeMIN = totalTimeSEC / 60
    totalTimeHR = totalTimeMIN / 60

    print(f"Total runtime was {totalTimeHR:.3f} hours, {totalTimeMIN:.3f} minutes, {totalTimeSEC:.3f} seconds")
    ops.wipe()
