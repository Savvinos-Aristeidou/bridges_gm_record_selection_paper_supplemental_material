from matplotlib import pyplot as plt
from openseespy import opensees as ops
import opsvis as opsv

from Units import MPa, GPa, m, g, mm, pi

# --------------------------------------
# Define materials
# --------------------------------------
# Steel reinforcement material
fy = 500.0 * MPa  # Yield strength
Es = 200.0 * GPa  # Elastic modulus
e_ult = 0.10  # Fracture strain of the steel

# Concrete materialz
fc = 42.0 * MPa  # Unconfined compressive strength
fu = 5.0 * MPa  # Unconfined compressive strength
eps_c = 0.002  # Strain at peak stress
eps_u = 0.010  # Strain at ultimate stress
k_conf = 1.2  # Confinement factor
Gc = 10.0 * GPa  # Shear modulus

# Unconfined
fc_unconf = fc
fu_unconf = fu
eps_c_unconf = eps_c
eps_u_unconf = eps_u

# Confined
fc_conf = k_conf * fc
fu_conf = k_conf * fu
eps_c_conf = eps_c * (1 + 5 * (k_conf - 1))
eps_u_conf = eps_u * (1 + 5 * (k_conf - 1))

# --------------------------------------
# Define sections
# -------------------------------------
# Deck section
Lspan = 50.0 * m  # Typical span length
Kpb = 26329  # Pot bearing stiffness
Ecd = 25.0 * GPa  # Elastic modulus of deck
Ad = 1.74e8 / Ecd  # Deck cross-sectional area
Izd = 1.34e8 / Ecd  # Deck second moment of area about z
Iyd = 2.21e9 / Ecd  # Deck second moment of area about y
Jd = 1.17e8 / Gc
rhod = 17.4  # Mass (tonnes) per meter length
DeckMass = rhod * g * Lspan / g  # Mass of 1 single deck span
DeckMass_half = 0.5 * DeckMass  # Mass of a half span

# Pier properties
Htyp = 7.0 * m  # Multiple for pier heights
Ecp = 25.0 * GPa  # Elastic modulus of pier concrete
bp = 2.0 * m  # Section width
hp = 4.0 * m  # Section height
tp = 0.4 * m  # Wall thickness
Jp = 2 * ((bp - tp) + (hp - tp)) * tp**3 / 3  # Torsional constant
cv = 20.0 * mm  # Concrete cover
Ap = hp * bp - (hp - 2 * tp) * (bp - 2 * tp)
rhop = 2.4 * Ap  # Mass per length of the pier

# Reinforcement properties
db = 20.0 * mm  # Bar diameter
Ab = 0.25 * pi * db**2  # Area of a single bar

# Fibre options
numFibreCoverTk = 5
numFibreCoverL = 50
numFibreCoreTk = 10
numFibreCoreL = 50

num_bars_flange = 18
num_bars_web = 7


class PierSections:
    def __init__(self, ElementFormulation):
        self.ElementFormulation = ElementFormulation
        return

    def define_sections(self):
        # --------------------------------------
        # Define element parameters
        # --------------------------------------
        # ElementFormulation coming from BridgeModel file
        if self.ElementFormulation == "FibreModel":
            ops.uniaxialMaterial("Steel02", 1, fy, Es, 0.005, 20, 0.925, 0.15)

            # Apply a min max limit to the steel
            ops.uniaxialMaterial("MinMax", 4, 1, '-min', -e_ult, '-max', e_ult)

            # Define uniaxial concrete materials
            # uniaxialMaterial Concrete01 $matTag $fpc    $epsc0        $fpcu       $epsU
            ops.uniaxialMaterial("Concrete01", 2, -fc_conf, -eps_c_conf,
                                -fu_conf, -eps_u_conf)
            ops.uniaxialMaterial("Concrete01", 3, -fc_unconf, -eps_c_unconf,
                                -fu_unconf, -eps_u_unconf)

            # Flange reinforcement layers (Material tag: 4)
            y_loc_flange_top = 0.5 * hp - cv - 0.5 * db
            y_loc_flange_bottom = 0.5 * hp - tp + cv + 0.5 * db
            y_loc_flange_negative_top = -0.5 * hp + cv + 0.5 * db
            y_loc_flange_negative_bottom = -0.5 * hp + tp - cv - 0.5 * db
            z_start_flange = 0.5 * bp - cv - 0.5 * db
            z_end_flange = -0.5 * bp + cv + 0.5 * db
            matTag_rebar = 4

            # Web reinforcement layers (Material tag: 4)
            x_start_web = -0.5 * hp + 2 * tp + 0.5 * db
            x_end_web = 0.5 * hp - 2 * tp - 0.5 * db
            z_loc_web_positive = 0.5 * bp - cv - 0.5 * db
            z_loc_web_inner_positive = 0.5 * bp - tp + cv + 0.5 * db
            z_loc_web_inner_negative = -0.5 * bp + tp - cv - 0.5 * db
            z_loc_web_negative = -0.5 * bp + cv + 0.5 * db

            # TODO: check if the way of defining the sections below is correct
            fib_sec_1 = [
                ['section', 'Fiber', 1, '-GJ', Gc * Jp],
                ['patch', "quad", 3, numFibreCoverL, numFibreCoverTk, -0.5 * hp, 0.5 * bp - cv, 0.5 * hp, 0.5 * bp - cv, 0.5 * hp, 0.5 * bp, -0.5 * hp, 0.5 * bp],
                ['patch', "quad", 3, numFibreCoverL, numFibreCoverTk, -0.5 * hp, -0.5 * bp, 0.5 * hp, -0.5 * bp, 0.5 * hp, -0.5 * bp + cv, -0.5 * hp, -0.5 * bp + cv],
                ['patch', "quad", 3, numFibreCoverTk, numFibreCoverL, -0.5 * hp, -0.5 * bp + cv, -0.5 * hp + cv, -0.5 * bp + cv, -0.5 * hp + cv, 0.5 * bp - cv, -0.5 * hp, 0.5 * bp - cv],
                ['patch', "quad", 3, numFibreCoverTk, numFibreCoverL, 0.5 * hp - cv, -0.5 * bp + cv, 0.5 * hp, -0.5 * bp + cv, 0.5 * hp, 0.5 * bp - cv, 0.5 * hp - cv, 0.5 * bp - cv],
                ['patch', "quad", 2, numFibreCoreL, numFibreCoreTk, -0.5 * hp + cv, 0.5 * bp - tp, 0.5 * hp - cv, 0.5 * bp - tp, 0.5 * hp - cv, 0.5 * bp - cv, -0.5 * hp + cv, 0.5 * bp - cv],
                ['patch', "quad", 2, numFibreCoreL, numFibreCoreTk, -0.5 * hp + cv, -0.5 * bp + cv, 0.5 * hp - cv, -0.5 * bp + cv, 0.5 * hp - cv, -0.5 * bp + tp, -0.5 * hp + cv, -0.5 * bp + tp],
                ['patch', "quad", 2, numFibreCoreTk, numFibreCoreL, -0.5 * hp + cv, -0.5 * bp + tp, -0.5 * hp + tp, -0.5 * bp + tp, -0.5 * hp + tp, 0.5 * bp - tp, -0.5 * hp + cv, 0.5 * bp - tp],
                ['patch', "quad", 2, numFibreCoreTk, numFibreCoreL, 0.5 * hp - tp, -0.5 * bp + tp, 0.5 * hp - cv, -0.5 * bp + tp, 0.5 * hp - cv, 0.5 * bp - tp, 0.5 * hp - tp, 0.5 * bp - tp],
                ['layer', 'straight', matTag_rebar, num_bars_flange, Ab, y_loc_flange_top, z_start_flange, y_loc_flange_top, z_end_flange],
                ['layer', 'straight', matTag_rebar, num_bars_flange, Ab, y_loc_flange_bottom, z_start_flange, y_loc_flange_bottom, z_end_flange],
                ['layer', 'straight', matTag_rebar, num_bars_flange, Ab, y_loc_flange_negative_top, z_start_flange, y_loc_flange_negative_top, z_end_flange],
                ['layer', 'straight', matTag_rebar, num_bars_flange, Ab, y_loc_flange_negative_bottom, z_start_flange, y_loc_flange_negative_bottom, z_end_flange],
                ['layer', 'straight', matTag_rebar, num_bars_web, Ab, x_start_web, z_loc_web_positive, x_end_web, z_loc_web_positive],
                ['layer', 'straight', matTag_rebar, num_bars_web, Ab, x_start_web, z_loc_web_inner_positive, x_end_web, z_loc_web_inner_positive],
                ['layer', 'straight', matTag_rebar, num_bars_web, Ab, x_start_web, z_loc_web_inner_negative, x_end_web, z_loc_web_inner_negative],
                ['layer', 'straight', matTag_rebar, num_bars_web, Ab, x_start_web, z_loc_web_negative, x_end_web, z_loc_web_negative],
            ]

            opsv.fib_sec_list_to_cmds(fib_sec_1)

            # matcolor = ['sandybrown', 'lightgrey', 'gold', 'w', 'w', 'w']  # Because there is an extra min/max material for concrete
            matcolor = ['r', 'lightgrey', 'gold', 'w', 'w', 'w']  # Because there is an extra min/max material for concrete
            opsv.plot_fiber_section(fib_sec_1, matcolor=matcolor)
            plt.axis('equal')
            # plt.title('Pier section - D = %.2f m, %dxΦ%.2f' % (D_pier, n_l_b, d_l_b * 1000))
            plt.title('Pier section')
            plt.xlabel('local z-coord [m]')
            plt.ylabel('local y-coord [m]')
            # plt.show()
            
            # Use a distributed plasticity element
            fiber_section = 1
            num_ips = 9
            integration1 = ["Lobatto", fiber_section, num_ips]  # 7m high pier
            integration2 = ["Lobatto", fiber_section, num_ips]  # 14m high pier
            integration3 = ["Lobatto", fiber_section, num_ips]  # 21m high pier
            
            # Apply integrations
            ops.beamIntegration('Lobatto', 1, fiber_section, num_ips)  # integration1
            ops.beamIntegration('Lobatto', 2, fiber_section, num_ips)  # integration2
            ops.beamIntegration('Lobatto', 3, fiber_section, num_ips)  # integration3
            
        elif self.ElementFormulation == "LumpedPlasticityModel":
            
            # Define your parameters
            Mzc = [20.0, 20.0, 20.0]
            Mzn = [46.0, 47.0, 48.0]
            Mzu = [51.8987, 53.0436, 54.2005]
            Mzr = [17.5, 20.0, 19.5]

            Myc = [8.0, 8.0, 8.0]
            Myn = [23.5, 24.0, 24.5]
            Myu = [25.1579, 25.7087, 26.2825]
            Myr = [10.0, 10.0, 10.0]

            phicz = [0.15, 0.15, 0.15]  # in rad/km for pier 1,2,3
            phinz = [1.25, 1.25, 1.25]
            phiuz = [26.9, 26.9, 26.9]
            phirz = [40, 60, 60]

            phicy = [0.25, 0.25, 0.25]
            phiny = [2.5, 2.5, 2.5]
            phiuy = [58.1, 58.1, 58.1]
            phiry = [110, 110, 110]

            # Create Elastic sections
            ops.section("Elastic", 101, Ecp, Ap, 1e3*Mzn[0]/phinz[0],
                        1e3*Myn[0]/phiny[0], Gc, Jp)
            ops.section("Elastic", 102, Ecp, Ap, 1e3*Mzn[1]/phinz[1],
                        1e3*Myn[1]/phiny[1], Gc, Jp)
            ops.section("Elastic", 103, Ecp, Ap, 1e3*Mzn[2]/phinz[2],
                        1e3*Myn[2]/phiny[2], Gc, Jp)

            # # Use the hysteretic material
            # # Create the hysteretic materials
            # uniaxialMaterial Hysteretic 101 [expr 1e3*[lindex $Mzn 0]] [expr [lindex $phinz 0]/1e3] [expr 1e3*[lindex $Mzu 0]] [expr [lindex $phiuz 0]/1e3] [expr 1e3*[lindex $Mzr 0]] [expr [lindex $phirz 0]/1e3] [expr -1e3*[lindex $Mzn 0]] [expr -[lindex $phinz 0]/1e3] [expr -1e3*[lindex $Mzu 0]] [expr -[lindex $phiuz 0]/1e3] [expr -1e3*[lindex $Mzr 0]] [expr -[lindex $phirz 0]/1e3] 0.5 0.5 0. 0. 0.
            # uniaxialMaterial Hysteretic 102 [expr 1e3*[lindex $Mzn 1]] [expr [lindex $phinz 1]/1e3] [expr 1e3*[lindex $Mzu 1]] [expr [lindex $phiuz 1]/1e3] [expr 1e3*[lindex $Mzr 1]] [expr [lindex $phirz 1]/1e3] [expr -1e3*[lindex $Mzn 1]] [expr -[lindex $phinz 1]/1e3] [expr -1e3*[lindex $Mzu 1]] [expr -[lindex $phiuz 1]/1e3] [expr -1e3*[lindex $Mzr 1]] [expr -[lindex $phirz 1]/1e3] 0.5 0.5 0. 0. 0.
            # uniaxialMaterial Hysteretic 103 [expr 1e3*[lindex $Mzn 2]] [expr [lindex $phinz 2]/1e3] [expr 1e3*[lindex $Mzu 2]] [expr [lindex $phiuz 2]/1e3] [expr 1e3*[lindex $Mzr 2]] [expr [lindex $phirz 2]/1e3] [expr -1e3*[lindex $Mzn 2]] [expr -[lindex $phinz 2]/1e3] [expr -1e3*[lindex $Mzu 2]] [expr -[lindex $phiuz 2]/1e3] [expr -1e3*[lindex $Mzr 2]] [expr -[lindex $phirz 2]/1e3] 0.5 0.5 0. 0. 0.
            #
            # uniaxialMaterial Hysteretic 201 [expr 1e3*[lindex $Myn 0]] [expr [lindex $phiny 0]/1e3] [expr 1e3*[lindex $Myu 0]] [expr [lindex $phiuy 0]/1e3] [expr 1e3*[lindex $Myr 0]] [expr [lindex $phiry 0]/1e3] [expr -1e3*[lindex $Myn 0]] [expr -[lindex $phiny 0]/1e3] [expr -1e3*[lindex $Myu 0]] [expr -[lindex $phiuy 0]/1e3] [expr -1e3*[lindex $Myr 0]] [expr -[lindex $phiry 0]/1e3] 0.5 0.5 0. 0. 0.
            # uniaxialMaterial Hysteretic 202 [expr 1e3*[lindex $Myn 1]] [expr [lindex $phiny 1]/1e3] [expr 1e3*[lindex $Myu 1]] [expr [lindex $phiuy 1]/1e3] [expr 1e3*[lindex $Myr 1]] [expr [lindex $phiry 1]/1e3] [expr -1e3*[lindex $Myn 1]] [expr -[lindex $phiny 1]/1e3] [expr -1e3*[lindex $Myu 1]] [expr -[lindex $phiuy 1]/1e3] [expr -1e3*[lindex $Myr 1]] [expr -[lindex $phiry 1]/1e3] 0.5 0.5 0. 0. 0.
            # uniaxialMaterial Hysteretic 203 [expr 1e3*[lindex $Myn 2]] [expr [lindex $phiny 2]/1e3] [expr 1e3*[lindex $Myu 2]] [expr [lindex $phiuy 2]/1e3] [expr 1e3*[lindex $Myr 2]] [expr [lindex $phiry 2]/1e3] [expr -1e3*[lindex $Myn 2]] [expr -[lindex $phiny 2]/1e3] [expr -1e3*[lindex $Myu 2]] [expr -[lindex $phiuy 2]/1e3] [expr -1e3*[lindex $Myr 2]] [expr -[lindex $phiry 2]/1e3] 0.5 0.5 0. 0. 0.

            # Function that creates Pinching4 materials
            def proc_uniaxial_pinching(materialTag, pEnvelopeStress, nEnvelopeStress, pEnvelopeStrain, nEnvelopeStrain, rDisp, rForce, uForce, gammaK, gammaD, gammaF, gammaE, damage):
                # ops.uniaxialMaterial("Pinching4", materialTag,
                #                      pEnvelopeStress[0], pEnvelopeStrain[0],
                #                      pEnvelopeStress[1], pEnvelopeStrain[1],
                #                      pEnvelopeStress[2], pEnvelopeStrain[2],
                #                      pEnvelopeStress[3], pEnvelopeStrain[3],
                #                      nEnvelopeStress[0], nEnvelopeStrain[0],
                #                      nEnvelopeStress[1], nEnvelopeStrain[1],
                #                      nEnvelopeStress[2], nEnvelopeStrain[2],
                #                      nEnvelopeStress[3], nEnvelopeStrain[3],
                #                      rDisp[0], rForce[0], uForce[0],
                #                      rDisp[1], rForce[1], uForce[1],
                #                      gammaK[0], gammaK[1], gammaK[2], gammaK[3], gammaK[4],
                #                      gammaD[0], gammaD[1], gammaD[2], gammaD[3], gammaD[4],
                #                      gammaF[0], gammaF[1], gammaF[2], gammaF[3], gammaF[4],
                #                      gammaE, damage)
                pinchX = 0.8  # 0.8
                pinchY = 0.2  # 0.2
                damage1 = 0.001
                damage2 = 0.0001
                ops.uniaxialMaterial("HystereticSM", materialTag,
                                     '-posEnv',
                                     pEnvelopeStress[0], pEnvelopeStrain[0],
                                     pEnvelopeStress[1], pEnvelopeStrain[1],
                                     pEnvelopeStress[2], pEnvelopeStrain[2],
                                     pEnvelopeStress[3], pEnvelopeStrain[3], 
                                     '-negEnv',
                                     nEnvelopeStress[0], nEnvelopeStrain[0],
                                     nEnvelopeStress[1], nEnvelopeStrain[1],
                                     nEnvelopeStress[2], nEnvelopeStrain[2],
                                     nEnvelopeStress[3], nEnvelopeStrain[3],
                                     '-pinch', pinchX, pinchY,
                                     '-damage', damage1, damage2,
                                     '-beta', 0)

            # Hysteretic parameters
            rDispV = [0.2, 0.2]  # Pos_env. Neg_env.##### Ratio of maximum deformation at which reloading begins
            rForceV = [-1.0, -1.0]  # Pos_env. Neg_env.##### Ratio of envelope force (corresponding to maximum deformation) at which reloading begins
            uForceV = [-1.0, -1.0]  # Pos_env. Neg_env.##### Ratio of monotonic strength developed upon unloading
            gammaKV = [0.0, 0.0, 0.0, 0.0, 0.0]  # gammaK1 gammaK2 gammaK3 gammaK4 gammaKLimit
            gammaDV = [0.0, 0.0, 0.0, 0.0, 0.0]  # gammaD1 gammaD2 gammaD3 gammaD4 gammaDLimit
            gammaFV = [0.0, 0.0, 0.0, 0.0, 0.0]  # gammaF1 gammaF2 gammaF3 gammaF4 gammaFLimit
            gammaEV = 0.0
            # gammaKV = [1.0, 0.2, 0.3, 0.2, 0.9]  # gammaK1 gammaK2 gammaK3 gammaK4 gammaKLimit
            # gammaDV = [0.5, 0.5, 2.0, 2.0, 0.5]  # gammaD1 gammaD2 gammaD3 gammaD4 gammaDLimit
            # gammaFV = [1.0, 0.0, 1.0, 1.0, 0.9]  # gammaF1 gammaF2 gammaF3 gammaF4 gammaFLimit
            # gammaEV = 10.0
            damV = "energy"

            # Bending moment points (in MNm)
            Mz1p = [1e3*Mzc[0], 1e3*Mzn[0], 1e3*Mzu[0], 1e3*Mzr[0]]  # stress1 stress2 stress3 stress4
            Mz2p = [1e3*Mzc[1], 1e3*Mzn[1], 1e3*Mzu[1], 1e3*Mzr[1]]
            Mz3p = [1e3*Mzc[2], 1e3*Mzn[2], 1e3*Mzu[2], 1e3*Mzr[2]]
            My1p = [1e3*Myc[0], 1e3*Myn[0], 1e3*Myu[0], 1e3*Myr[0]]  # stress1 stress2 stress3 stress4
            My2p = [1e3*Myc[1], 1e3*Myn[1], 1e3*Myu[1], 1e3*Myr[1]]
            My3p = [1e3*Myc[2], 1e3*Myn[2], 1e3*Myu[2], 1e3*Myr[2]]
            Mz1n = [-1e3*Mzc[0], -1e3*Mzn[0], -1e3*Mzu[0], -1e3*Mzr[0]]  # stress1 stress2 stress3 stress4
            Mz2n = [-1e3*Mzc[1], -1e3*Mzn[1], -1e3*Mzu[1], -1e3*Mzr[1]]
            Mz3n = [-1e3*Mzc[2], -1e3*Mzn[2], -1e3*Mzu[2], -1e3*Mzr[2]]
            My1n = [-1e3*Myc[0], -1e3*Myn[0], -1e3*Myu[0], -1e3*Myr[0]]  # stress1 stress2 stress3 stress4
            My2n = [-1e3*Myc[1], -1e3*Myn[1], -1e3*Myu[1], -1e3*Myr[1]]
            My3n = [-1e3*Myc[2], -1e3*Myn[2], -1e3*Myu[2], -1e3*Myr[2]]

            phiz1p = [phicz[0]/1e3, phinz[0]/1e3, phiuz[0]/1e3, phirz[0]/1e3]  # strain1 strain2 strain3 strain4
            phiz2p = [phicz[1]/1e3, phinz[1]/1e3, phiuz[1]/1e3, phirz[1]/1e3]
            phiz3p = [phicz[2]/1e3, phinz[2]/1e3, phiuz[2]/1e3, phirz[2]/1e3]
            phiy1p = [phicy[0]/1e3, phiny[0]/1e3, phiuy[0]/1e3, phiry[0]/1e3]  # strain1 strain2 strain3 strain4
            phiy2p = [phicy[1]/1e3, phiny[1]/1e3, phiuy[1]/1e3, phiry[1]/1e3]
            phiy3p = [phicy[2]/1e3, phiny[2]/1e3, phiuy[2]/1e3, phiry[2]/1e3]
            phiz1n = [-phicz[0]/1e3, -phinz[0]/1e3, -phiuz[0]/1e3, -phirz[0]/1e3]  # strain1 strain2 strain3 strain4
            phiz2n = [-phicz[1]/1e3, -phinz[1]/1e3, -phiuz[1]/1e3, -phirz[1]/1e3]
            phiz3n = [-phicz[2]/1e3, -phinz[2]/1e3, -phiuz[2]/1e3, -phirz[2]/1e3]
            phiy1n = [-phicy[0]/1e3, -phiny[0]/1e3, -phiuy[0]/1e3, -phiry[0]/1e3]  # strain1 strain2 strain3 strain4
            phiy2n = [-phicy[1]/1e3, -phiny[1]/1e3, -phiuy[1]/1e3, -phiry[1]/1e3]
            phiy3n = [-phicy[2]/1e3, -phiny[2]/1e3, -phiuy[2]/1e3, -phiry[2]/1e3]

            proc_uniaxial_pinching(101, Mz1p, Mz1n, phiz1p, phiz1n, rDispV, rForceV, uForceV, gammaKV, gammaDV, gammaFV, gammaEV, damV)
            proc_uniaxial_pinching(102, Mz2p, Mz2n, phiz2p, phiz2n, rDispV, rForceV, uForceV, gammaKV, gammaDV, gammaFV, gammaEV, damV)
            proc_uniaxial_pinching(103, Mz3p, Mz3n, phiz3p, phiz3n, rDispV, rForceV, uForceV, gammaKV, gammaDV, gammaFV, gammaEV, damV)

            proc_uniaxial_pinching(201, My1p, My1n, phiy1p, phiy1n, rDispV, rForceV, uForceV, gammaKV, gammaDV, gammaFV, gammaEV, damV)
            proc_uniaxial_pinching(202, My2p, My2n, phiy2p, phiy2n, rDispV, rForceV, uForceV, gammaKV, gammaDV, gammaFV, gammaEV, damV)
            proc_uniaxial_pinching(203, My3p, My3n, phiy3p, phiy3n, rDispV, rForceV, uForceV, gammaKV, gammaDV, gammaFV, gammaEV, damV)

            # Create the elements (assuming 1D elements for demonstration) -> this is not included in the tcl version
            # ops.element("FiberSection", 101, *integration1)
            # ops.element("FiberSection", 102, *integration2)
            # ops.element("FiberSection", 103, *integration3)

            # Define Uniaxial Sections
            # section('Uniaxial', secTag, matTag, quantity)
            ops.section("Uniaxial", 201, 101, "Mz")
            ops.section("Uniaxial", 202, 102, "Mz")
            ops.section("Uniaxial", 203, 103, "Mz")

            # Aggregate Myy behavior to Mzz behavior
            ops.section("Aggregator", 301, 201, "My", "-section", 201)
            ops.section("Aggregator", 302, 202, "My", "-section", 202)
            ops.section("Aggregator", 303, 203, "My", "-section", 203)

            # Calculate hinge lengths
            Lp1 = 0.05 * Htyp * 1 + 0.022 * db * fy / MPa  # Use 0.05 because of the min(0.8,0.2*(fu/fy-1))
            Lp2 = 0.05 * Htyp * 2 + 0.022 * db * fy / MPa
            Lp3 = 0.05 * Htyp * 3 + 0.022 * db * fy / MPa

            # Apply integrations
            ops.beamIntegration('HingeRadau', 1, 301, Lp1, 301, Lp1, 101)  # integration1
            ops.beamIntegration('HingeRadau', 2, 302, Lp2, 302, Lp2, 102)  # integration1
            ops.beamIntegration('HingeRadau', 3, 303, Lp3, 303, Lp3, 103)  # integration1

        return
