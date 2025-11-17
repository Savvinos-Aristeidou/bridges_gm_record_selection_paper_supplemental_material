import numpy as np
import openseespy.opensees as ops
from pathlib import Path
from ReadGMFile import ReadGMFile

class Analysis:
    g = 9.81

    def __init__(self):
        """_summary_
        """

    def perform_cyclic_push(self, dref=0.5, mu=1, numCycles=6, ctrlNode=2, dispDir=1, dispIncr=200, IOflag=1, PrintFlag=1):
        """Procedure to carry out a cyclic pushover of a model
        Adapted from Gerard O'Reilly's CyclicPush Routine
        --------------------------------------------------
        Description of Parameters
        --------------------------------------------------
        Command:      CyclicPush
        dref:         Reference displacement. Corresponds to yield or equivalent other.
        mu:           Multiple of dref to which the cycles is run.
        numCycles:    No. of cycles. Valid options either 1,2,3,4,5,6
        ctrlNode:     Node to control with the displacement integrator
        dispDir:      Direction the loading is applied.
        dispIncr:     Number of displacement increments.
        IOflag:       Option to print cycle details on screen. 1 for on, 0 for off
        PrintFlag:    Optional flag for printing nodes/elements at max cycle
        ---------------------------------------------------
        """
        ops.wipeAnalysis()
        # ops.timeSeries('Linear', 2)  # define the timeSeries for the load pattern
        # ops.pattern('Plain', 2, 2)  # define load pattern -- generalized
        # ops.load(2, *[1, 0, 0])

        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('FullGeneral')

        LoadFactor = [0]
        # DispCtrlNode = [0]
        Disp1CtrlNode = [0]
        Disp2CtrlNode = [0]
        testType = {1: 'NormUnbalance', 2: 'NormDispIncr', 3: 'EnergyIncr', 4: 'RelativeNormUnbalance',
                    5: 'RelativeNormDispIncr', 6: 'RelativeTotalNormDispIncr', 7: 'RelativeEnergyIncr', 8: 'FixedNumIter'}

        # Algorithm Types
        algoType = {1: 'Newton', 2: 'ModifiedNewton', 3: 'KrylovNewton', 4: 'Broyden', 6: 'ModifiedNewton',
                    7: 'NewtonLineSearch'}

        # Integrator Types
        intType = {1: 'DisplacementControl', 2: 'LoadControl', 3: 'Parallel DisplacementControl',
                4: 'MinUnbalDispNorm', 5: 'Arc-Length Control'}

        tolInit = 1.0e-5  # Set the initial Tolerance, so it can be referred back to
        iterInit = 300  # Set the initial Max Number of Iterations

        # test(testType, *testArgs)    e.g. test('NormUnbalance', tol, iter, pFlag=0, nType=2, maxincr=-1)
        ops.test(testType[1], tolInit, iterInit)

        # algorithm(algoType, *algoArgs)   e.g. algorithm('Newton', secant=False, initial=False, initialThenCurrent=False)
        ops.algorithm(algoType[1])

        # Create the list of displacements
        if numCycles == 0:
            dispList = [dref * mu]
        else:
            dispList = [j * dref * mu * i / numCycles for i in range(1, numCycles + 1) for j in [1, -2, 1]]

        dispNoMax = len(dispList)

        # Print values
        if IOflag == 1:
            print("CyclicPush: %s cycles to mu = %s at %s" % (numCycles, mu, ctrlNode))

        # Carry out loading
        for d in range(1, dispNoMax + 1, 1):
            numIncr = dispIncr
            dU = dispList[d - 1] / (1.0 * numIncr)
            ops.integrator(intType[1], ctrlNode, dispDir, dU)
            # ops.integrator(intType[1], ctrlNode, dispDir[1], dU)
            # ops.integrator('LoadControl', 1)
            ops.analysis('Static')
            

            for l in range(0, numIncr, 1):
                # print("Analysis step:", l)
                ok = ops.analyze(1)
                
                # LoadFactor.append(ops.getTime())
                LoadFactor.append(-ops.eleForce(1, dispDir))

                Disp1CtrlNode.append(ops.nodeDisp(ctrlNode, dispDir))
                # Disp2CtrlNode.append(ops.nodeDisp(ctrlNode, dispDir[1]))

                if ok != 0:
                    print("DispControl Analysis is FAILED")
                    print("Analysis failed at cycle: %s and dispIncr: %s" % (d, l))
                    print('-------------------------------------------------------------------------')
                    break

            if PrintFlag == 1 and d == 1:
                ops.printModel('-file', 'nodes_' + str(mu) + '.txt', '-node')
                ops.printModel('-file', 'ele_' + str(mu) + '.txt', '-ele')
        if ok == 0:
            print("DispControl Analysis is SUCCESSFUL")
            print('-------------------------------------------------------------------------')
        # DispCtrlNode = np.sqrt(np.array(Disp1CtrlNode)**2+np.array(Disp2CtrlNode)**2)
        DispCtrlNode = np.array(Disp1CtrlNode)

        LoadFactor = np.array(LoadFactor)
        DispCtrlNode = np.array(DispCtrlNode)
        
        return LoadFactor, DispCtrlNode

    def perform_NRHA(self, GMdatabase_dir, filename, dampRatio, omega, SF, Dc, tNode, bNode, pier_ele):

        # dt1, nPts1, acc1 = ReadGMFile(inFile=GMdatabase_dir / filename)
        dt2, nPts2, acc2 = ReadGMFile(inFile=GMdatabase_dir / filename)
        # dt2, nPts2, acc2 = ReadGMFile(inFile=GMdatabase_dir / meta_data['Filename_2'][GM_idx])
        # nPts = np.min([nPts1, nPts2])
        # acc1 = acc1[:nPts]
        # acc2 = acc2[:nPts]

        #  ----------------------------------------------------------------------------
        #  Nonlinear Response History Analysis
        #  ----------------------------------------------------------------------------
        ops.wipeAnalysis()
        # --------------------------------------
        # LATERAL ANALYSIS PARAMETERS
        # --------------------------------------
        ops.constraints('Transformation')
        ops.numberer('RCM')
        ops.system('UmfPack')  # changed from 'full general'

        # --------------------------------------
        # DEFINE DAMPING
        # --------------------------------------
        w1 = omega[0]
        w2 = omega[2]
        a0 = 2 * w1 * w2 / (w2 * w2 - w1 * w1) * (w2 * dampRatio - w1 * dampRatio)
        a1 = 2 * w1 * w2 / (w2 * w2 - w1 * w1) * (-dampRatio / w2 + dampRatio / w1)
        #   rayleigh(alphaM, betaK, betaKinit, betaKcomm)
        ops.rayleigh(a0, 0, 0, a1)

        #  timeSeries('Path', tag, '-dt', dt, '-values', *values, '-factor', factor=1.0)
        # ops.timeSeries('Path', 2, '-dt', dt1, '-values', *acc1, '-factor', self.g * SF)
        ops.timeSeries('Path', 3, '-dt', dt2, '-values', *acc2, '-factor', self.g * SF)

        # pattern('UniformExcitation', patternTag, dir, '-accel', accelSeriesTag)
        # ops.pattern('UniformExcitation', 2, 1, '-accel', 2)
        ops.pattern('UniformExcitation', 3, 2, '-accel', 3)

        # NRtype = "EnvelopeNode"
        # ERtype = "EnvelopeElement"
        # # Displacements
        # ops.recorder(NRtype, '-file',
        #             f"{outsdir}/DeckDispX.{AType}.{self.model_num}.txt",
        #             '-node', deck_node, '-dof', 1, 'disp')
        # ops.recorder(NRtype, '-file',
        #             f"{outsdir}/DeckDispY.{AType}.{self.model_num}.txt",
        #             '-node', deck_node, '-dof', 2, 'disp')
        # # Base reaction
        # ops.recorder(NRtype, '-file', f"{outsdir}/rxnX.{AType}.{self.model_num}.txt",
        #             '-node', base_node, '-dof', 1, 'reaction')
        # ops.recorder(NRtype, '-file', f"{outsdir}/rxnY.{AType}.{self.model_num}.txt",
        #             '-node', base_node, '-dof', 2, 'reaction')
        # # Pier elements
        # ops.recorder(ERtype, '-file',
        #             f"{outsdir}/PierSectionForce.{AType}.{self.model_num}.txt",
        #             '-ele', pier_ele, 'section', 'force')
        # ops.recorder(ERtype, '-file',
        #             f"{outsdir}/PierSectionDeformation.{AType}.{self.model_num}.txt",
        #             '-ele', pier_ele, 'section', 'deformation')

        # TODO: change drift with maximum curvature from all piers
        curvature_max, disp_x, disp_y, Mdrft, cIndex, mflr, Analysis, accel_x, accel_y, time = self._NRHA(dt2, dt2 * nPts2, Dc, tNode, bNode, '', pier_ele, pflag=0)

        ops.wipe()

        return curvature_max

    def _NRHA(self, Dt, Tmax, Dc, tNode, bNode, log, pier_ele, pflag=0):
        # ----------------------------------------------------------------
        # -- Script to Conduct 3D Non-linear Response History Analysis ---
        # ----------------------------------------------------------------
        #
        # This procedure is a simple script that executes the NRHA of a 3D model. It
        # requires that the model has the dynamic analysis objects defined and just the
        # 'analyze' of a regular OpenSees model needs to be executed. Therefore, the model
        # needs to be built to the point where a modal analysis can be conducted. The
        # ground motion timeSeries and pattern need to be setup and the constraints,
        # numberer and system analysis objects also need to be predefined.
        #   
        # When conducting the NRHA, this proc will try different options to achieve
        # convergence and finish the ground motion. This allows for a relatively robust
        # analysis algorithm to be implemented with a single command.
        #
        # In addition, the analysis requires that a deformation capacity be specified
        # to define a limit that upon exceedance, will stop the NRHA and mark the
        # analysis as a collapse case. It monitors the current deformation of a number
        # of specified nodes and flags collapse based on their deformation. This
        # prevents the models getting 'stuck' trying to converge a model that has
        # essentially collapsed, or would be deemed a collapse case in post processing.
        # These are defined in terms of the pier drifts. For 3D analysis, the SRSS 
        # of absolute maximum drift in either direction is used. 
        # Other definitions are possible but not yet implemented.
        #
        # Lastly, a log file identifier is also required in the input. This will log
        # all of the essential information about the maximum pier drifts. This script
        # was developed for analysing bridges so the deformation capacity typically
        # corresponds to a drift capacity and the top and bottom nodes would typically
        # correspond to the centreline nodes of the bridge pier nodes.
        #
        # --------------------------------------------------
        # Description of Parameters
        # --------------------------------------------------
        # Dt:       Analysis time step
        # Tmax:     Length of the record (including padding of 0's)
        # Dc:       Drift capacity for pier drift (%)
        # tNode:    List of top nodes (e.g. [2, 3, 4, 5])
        # bNode:    List of bottom node (e.g. [1, 2, 3, 4])
        # log:      File handle of the logfile
        # pflag:    Flag to print stuff if necessary
        # --------------------------------------------------

        # Define the Initial Analysis Parameters
        testType = 'NormDispIncr'        # Set the initial test type (default)
        tolInit = 1.0e-7                 # Set the initial Tolerance, so it can be referred back to (default)
        iterInit = 20                    # Set the initial Max Number of Iterations (default)
        algorithmType = 'KrylovNewton'   # Set the initial algorithm type (default)
        # Parameters required in Newmark Integrator
        gamma = 0.5; beta = 0.25 # gamma = 1/2, beta = 1/4 --> Average Acceleration Method; # gamma = 1/2, beta = 1/6 --> Linear Acceleration Method;
        # Parameters required in Hilber-Hughes-Taylor (HHT) integrator
        alpha = 0.85 # alpha = 1.0 = Newmark Method. smaller alpha means greater numerical damping. 0.67<alpha<1.0 recommended. Leave beta and gamma as default for unconditional stability.

        # Set up analysis parameters
        cIndex = 0              # Initially define the control index (-1 for non-converged, 0 for stable, 1 for global collapse)
        controlTime = 0.0       # Start the controlTime
        ok = 0                  # Set the convergence to 0 (initially converged)
        mflr = 0                # Set the initial pier collapse location
        Mdrft = 0.0             # Set initially the maximum of all pier drifts (SRSS)
        time = []
        # Set up the pier drift and acceleration values
        h = []
        mdrftX = []
        mdrftY = []
        mdrft = []
        
        # initiate displacement time histories for SDOF
        disp_x = []
        disp_y = []
        
        accel_x = []
        accel_y = []
        
        curvatures_all = []
        
        for i in range(len(tNode)):
            # Find the coordinates of the nodes in Global Z (3)
            top2 = ops.nodeCoord(tNode[i], 2)
            bot2 = ops.nodeCoord(bNode[i], 2)
            dist = top2-bot2
            
            if dist == 0: dist = 1
            
            # This means we take the distance in Z (3) in my coordinates systems at least. This is X-Y/Z| so X=1 Y=2 Z=3. (gli altri vacca gare)
            h.append(dist);         # Current pier height
            mdrftX.append(0.0)      # We will populate the lists with zeros initially
            mdrftY.append(0.0)
            mdrft.append(0.0)            
            if dist==0: print("WARNING: Zerolength found in drift check")
        
        
        # Run the actual analysis now
        while cIndex==0 and controlTime <= Tmax and ok==0:
            # Set the default analysis parameters
            ops.integrator('Newmark',gamma,beta)
            ops.test(testType, tolInit, iterInit)
            ops.algorithm(algorithmType)
            ops.analysis('Transient')

            # Do the analysis
            ok = ops.analyze(1, Dt)        # Run a step of the analysis
            controlTime = ops.getTime()    # Update the control time
            time.append(controlTime)
            if pflag>1: print("Completed %.2f of %.2f seconds" % (controlTime, Tmax))

            # If the analysis fails, try the following changes to achieve convergence
            # Analysis will be slower in here though...
            k = 0 # counter for the integrators
            while k < 2 and ok!=0:
                if k == 1:
                    ops.integrator('HHT',alpha) # Hail Mary... Bless the analysis with thy damping!
                    print(" ~~~ Changing integrator to Hilber-Hughes-Taylor (HHT) from Newmark at %.2f......" % controlTime)
                # First changes are to change algorithm to achieve convergence...
                if ok != 0:
                    print(" ~~~ Failed at %.2f - Reduced timestep by half......" % controlTime)
                    Dtt = 0.5*Dt
                    ok = ops.analyze(1, Dtt)
                    
                if ok != 0:
                    print(" ~~~ Failed at %.2f - Reduced timestep by quarter......" % controlTime)
                    Dtt = 0.25*Dt
                    ok = ops.analyze(1, Dtt)
                    
                if ok != 0:
                    print(" ~~~ Failed at %.2f - Trying Broyden......" % controlTime)
                    ops.algorithm('Broyden', 8)
                    ok = ops.analyze(1, Dt)

                if ok != 0:
                    print(" ~~~ Failed at %.2f - Trying Newton with Initial Tangent......" % controlTime)
                    ops.algorithm('Newton', '-initial')
                    ok = ops.analyze(1, Dt)

                if ok != 0:
                    print("Failed at %.2f - Trying NewtonWithLineSearch......" % controlTime)
                    ops.algorithm('NewtonLineSearch', 0.8)
                    ok = ops.analyze(1, Dt)

                # Next change both algorithm and tolerance to achieve convergence....
                if ok != 0:
                    print(" ~~~ Failed at %.2f - Trying Broyden & relaxed convergence......" % controlTime)
                    ops.test(testType, tolInit*0.1, iterInit*50)
                    ops.algorithm('Broyden', 8)
                    ok = ops.analyze(1, Dt)

                if ok != 0:
                    print(" ~~~ Failed at %.2f - Trying Newton with Initial Tangent & relaxed convergence......" % controlTime)
                    ops.test(testType, tolInit*0.1, iterInit*50)
                    ops.algorithm('Newton', '-initial')
                    ok = ops.analyze(1, Dt)

                if ok != 0:
                    print(" ~~~ Failed at %.2f - Trying NewtonWithLineSearch & relaxed convergence......" % controlTime)
                    ops.test(testType, tolInit*0.1, iterInit*50)
                    ops.algorithm('NewtonLineSearch', 0.8)
                    ok = ops.analyze(1, Dt)

                # Next half the timestep with both algorithm and tolerance reduction, if this doesn't work - in bocca al lupo
                if ok != 0:
                    print(" ~~~ Failed at %.2f - Trying Broyden, reduced timestep & relaxed convergence......" % controlTime)
                    ops.test(testType, tolInit*0.1, iterInit*50)
                    ops.algorithm('Broyden', 8)
                    Dtt = 0.5*Dt
                    ok = ops.analyze(1, Dtt)

                if ok != 0:
                    print(" ~~~ Failed at %.2f - Trying Newton with Initial Tangent, reduced timestep & relaxed convergence......"  % controlTime)
                    ops.test(testType, tolInit*0.1, iterInit*50)
                    ops.algorithm('Newton', '-initial')
                    Dtt = 0.5*Dt
                    ok = ops.analyze(1, Dtt)

                if ok != 0:
                    print(" ~~~ Failed at %.2f - Trying NewtonWithLineSearch, reduced timestep & relaxed convergence......" % controlTime)
                    ops.test(testType, tolInit*0.1, iterInit*50)
                    ops.algorithm('NewtonLineSearch', 0.8)
                    Dtt = 0.5*Dt
                    ok = ops.analyze(1, Dtt)
                k += 1

            # Shit...  Failed to converge, exit the analysis.
            if ok !=0:
                print(" ~~~ Failed at %.2f - exit the analysis......" % controlTime)
                ops.wipe()
                cIndex = -1

            if ok == 0:

                # Check the pier drifts
                for i in range(len(tNode)):

                    accel_x.append(ops.nodeAccel(tNode[i], 1)+ops.nodeAccel(bNode[i], 1))
                    accel_y.append(ops.nodeAccel(tNode[i], 2)+ops.nodeAccel(bNode[i], 2))

                    # SDOF displacement time history
                    disp_x.append(ops.nodeDisp(tNode[i], 1))
                    disp_y.append(ops.nodeDisp(tNode[i], 2))

                    tNode_dispX = ops.nodeDisp(tNode[i], 1)                               # Current top node disp in X
                    tNode_dispY = ops.nodeDisp(tNode[i], 2)                               # Current top node disp in Y
                    bNode_dispX = ops.nodeDisp(bNode[i], 1)                               # Current bottom node disp in X
                    bNode_dispY = ops.nodeDisp(bNode[i], 2)                               # Current bottom node disp in Y
                    cHt    = h[i]                                                        # Current pier height
                    cdrftX = abs(tNode_dispX-bNode_dispX)/cHt                            # Current pier drift ratio in X at the current pier
                    cdrftY = abs(tNode_dispY-bNode_dispY)/cHt                            # Current pier drift ratio in Y at the current pier
                    cdrft =    ((cdrftX**2)+(cdrftY**2))**0.5                            # SRSS of two drift components
                    if cdrftX >= mdrftX[i]: mdrftX[i] = cdrftX
                    if cdrftY >= mdrftY[i]: mdrftY[i] = cdrftY
                    if cdrft >= mdrft[i]: mdrft[i] = cdrft
                    if cdrft > Mdrft: Mdrft = cdrft; mflr = i+1                          # Update the current maximum pier drift and where it is

                    curvatures_all.append(ops.eleResponse(pier_ele[i], 'section', 6, 'deformation')[0])

                # if Mdrft >= Dc: 
                #     cIndex = 1
                #     Mdrft = Dc
                #     ops.wipe()                        # Set the state of the model to local collapse (=1)

        curvature_max = max(np.abs(curvatures_all))
        if cIndex == -1:
            Analysis = "Analysis is FAILED to converge at %.3f of %.3f" % (controlTime, Tmax)
        if cIndex == 0:
            # Analysis = "Analysis is SUCCESSFULLY completed\nPeak Pier Drift: %.2f%% at Pier %d" % (Mdrft, mflr)
            # Analysis = "Analysis is SUCCESSFULLY completed\nPeak Pier Drift: %.2f%%" % Mdrft
            Analysis = "Analysis is SUCCESSFULLY completed\nPeak Displacement: %.4fm" % (Mdrft)
        if cIndex == 1:
            Analysis = "Analysis is STOPPED, peak column drift ratio, %d%%, is exceeded, local COLLAPSE is observed" % Dc

        if pflag != 0:
            print(Analysis)

        if pflag>0:
            # Create some output
            f = open(log,"w+")
            f.write(Analysis+'\n')   # Print to the logfile the analysis state

            # Print to the max interpier drifts
            f.write("Peak Pier DriftX: ")
            for i in range(len(mdrftX)):
                f.write("%.2f " % mdrftX[i])
            f.write("%\n")
            f.write("Peak Pier DriftY: ")
            for i in range(len(mdrftY)):
                f.write("%.2f " % mdrftY[i])
            f.write("%\n")
            f.write("Peak Pier Drift: ")
            for i in range(len(mdrft)):
                f.write("%.2f " % mdrft[i])
            f.write("%")
            f.close()
            
        return curvature_max, disp_x ,disp_y ,Mdrft, cIndex, mflr, Analysis, accel_x, accel_y, time

