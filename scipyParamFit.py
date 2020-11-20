#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 18:27:10 2020

@author: michal
"""

from os import mkdir, getcwd, chdir, system, remove
from os.path import isdir, abspath, join
from shutil import copyfile
import pandas as pd
import numpy as np
from glob import glob
from scipy.optimize import minimize
from parmed.amber import AmberParm

class Parameter2Optimize:
    def __init__(self, kind, valueType, bounds, atomMask):
        self.kind = kind
        self.valueType = valueType
        self.bounds = bounds
        self.atomMask = atomMask

class ParameterOptimizer:
    def __init__(self, prmtopInit , referenceTrajectory, referenceTrajectoryForCppTraj, evaluationCriteria, 
                 referenceEnergies, sanderMinimization, sanderMD, MDseed,
                 parameters2optimizeFile):
        self.initPrmtop = abspath(prmtopInit)
        self.referenceTrajectory = abspath(referenceTrajectory)
        self.referenceTrajectoryCppTraj = abspath(referenceTrajectoryForCppTraj)
        
        self.evaluationCriteria = abspath(evaluationCriteria)
        self.referenceEnegiesFile = abspath(referenceEnergies)
        
        self.scratchDir = join( getcwd(),  "scr")
        
        if not isdir(self.scratchDir):
            mkdir(self.scratchDir)
        else:
            self.cleanScratch()
            
            
        self.mdSeed = abspath( MDseed )
        self.sanderMinFile = abspath( sanderMinimization )
        self.sanderMDFile = abspath( sanderMD )
        self.referenceMeans = self.getReferenceMeans()
        self.parameter2optimize = []
        self.readParameters2optimize(parameters2optimizeFile)
        self.evaluationKinds = self.getKindOfEvaluationParameters()
        
        self.currentPrmtop = abspath( "current.prmtop")
        copyfile( self.initPrmtop, self.currentPrmtop )
        
        self.evaluationMDtrajectory = join( self.scratchDir, "evaluationMD.nc" )
        
    def cleanScratch(self):
        for f in glob( join(self.scratchDir, "*" ) ):
            remove(f)
        
    def getReferenceMeans(self):
        return self.evaluateWithCppTraj( self.referenceTrajectoryCppTraj )
    
    def getBounds(self):
        return [ p.bounds for p in self.parameter2optimize ]
    
    def getKindOfEvaluationParameters(self):
        evCriteriaF = open(self.evaluationCriteria, 'r')
        line = evCriteriaF.readline()
        header2kind ={}
        while line:
            lineS = line.split()
            header2kind[lineS[1]] = lineS[0]
            
            line = evCriteriaF.readline()
        
        evCriteriaF.close()
        
        return header2kind
        
    def evaluateWithCppTraj(self, trajectory ):
        referenceInputName = "evaluation.cpptraj"
        cppTrajInp = open(join(self.scratchDir, referenceInputName ), 'w')
        cppTrajInp.write( "trajin "+trajectory+"\n" )
        
        evCriteriaF = open(self.evaluationCriteria, 'r')
        line = evCriteriaF.readline()
        headers = []
        files = []
        
        while line:
            lineS = line.split()
            headers.append(lineS[1])
            files.append( lineS[-1] )
            
            
            cppTrajInp.write(line)
            line = evCriteriaF.readline()
        
        evCriteriaF.close()
        cppTrajInp.write("\n")
        cppTrajInp.close()
        
        
        currentDir = getcwd()
        chdir(self.scratchDir)
        system( "cpptraj -p "+self.initPrmtop+" -i "+referenceInputName+" > reference.log"  )
        
        header2meanValue = {}
        for h, f in zip(headers, files):
            header2meanValue[h] = np.mean(pd.read_csv( f, sep='\s+'  )[ h ])
        
        chdir(currentDir)
        return header2meanValue
    
    def readParameters2optimize(self, paramsFile):
        pf = open(paramsFile, 'r')
        line = pf.readline()
        
        state = "idle"
        while line:
            if "# BOND INFORMATION:" in line:
                state = "readBonds"
            elif "# ANGLE PARAMETERS:" in line:
                state = "readAngles"
            elif "# DIHEDRAL PARAMETERS:" in line:
                state = "readDihedral"
                
                
            if line.strip()[0] != "#" :
                lineS = line.split()
                if state == "readBonds":
                    atomMask = set(lineS[:2])
                    for flag, valueType, bounds in zip(lineS[-2:], ["req", "kr"], [ (0.001, 4), (0.001, 1000) ] ):
                        if flag == "1":
                            self.parameter2optimize.append( Parameter2Optimize("bond", valueType, bounds, atomMask) )
                    
                elif state == "readAngles":
                    atomMask = lineS[:3]
                    for flag, valueType, bounds in zip(lineS[-2:], ["kt", "theq"], [ (0.001, 200), (0.001, 180) ] ):
                        if flag == "1":
                            self.parameter2optimize.append( Parameter2Optimize("angle", valueType, bounds, atomMask) )
                            
#                elif state == "readDihedrals":
#                    atomMask = lineS[:4]
#                    for flag, valueType, bounds in zip(lineS[-4:], ["term", "kp", "np", "phase" ], [ (0.001, 100), (0.001, 180) ] ):
#                        if flag == "1":
#                            self.parameter2optimize.append( Parameter2Optimize("angle", valueType, bounds, atomMask) )
            
            line = pf.readline()
        
        pf.close()
        
        
    def readX(self):
        currentParams = AmberParm(self.currentPrmtop)
        x = []
        
        for p in self.parameter2optimize:
            if p.kind == "bond":
                for b in currentParams.bonds:
                    if set( [b.atom1.type, b.atom2.type]) == p.atomMask:
                        if p.valueType == "req":
                            value = b.type.req
                        elif p.valueType == "kr":
                            value = b.type.k
                            
                        x.append(value)
                        break
                    
            elif p.kind == "angle":
                for a in currentParams.angles:
                    angleTypes = [ a.atom1.type, a.atom2.type, a.atom3 ]
                    angleTypesRev = reversed( angleTypes )
                    if  angleTypes == p.atomMask or angleTypesRev == p.atomMask:
                        if p.valueType == "kt":
                            value = a.type.k
                        elif p.valueType == "theq":
                            value = a.type.theteq
                            
                        x.append(value)
                        break

        return x
    
    def setX(self, xVec):
        currentParams = AmberParm(self.currentPrmtop)
        
        for x, p in zip( xVec, self.parameter2optimize):
            if p.kind == "bond":
                for b in currentParams.bonds:
                    if set( [b.atom1.type, b.atom2.type]) == p.atomMask:
                        if p.valueType == "req":
                            b.type.req = x
                        elif p.valueType == "kr":
                            b.type.k = x
                            
                        break
                    
            elif p.kind == "angle":
                for a in currentParams.angles:
                    angleTypes = [ a.atom1.type, a.atom2.type, a.atom3 ]
                    angleTypesRev = reversed( angleTypes )
                    if  angleTypes == p.atomMask or angleTypesRev == p.atomMask:
                        if p.valueType == "kt":
                            a.type.k = x
                        elif p.valueType == "theq":
                            a.type.theteq = x
                            
                        break
        
        currentParams.write_parm( self.currentPrmtop )
    
    def evaluateWithParamFit(self):
        strNo = np.loadtxt( self.referenceEnegiesFile ).size
        
        paramfitInputFilename = join(self.scratchDir, "paramfit1.in") 
        paramfitInput = open( paramfitInputFilename , 'w' )
        paramfitInput.write( """RUNTYPE=FIT
NSTRUCTURES={strNo}
COORDINATE_FORMAT=TRAJECTORY
PARAMETERS_TO_FIT=K_ONLY
FITTING_FUNCTION=SIMPLEX

""".format( strNo = strNo ) )
        paramfitInput.close()
        
        paramfitOutFilename = join(self.scratchDir, "paramfit1.out")
        system( "paramfit -i "+paramfitInputFilename+" -p " + self.currentPrmtop + " -c " + self.referenceTrajectory + " -q " + self.referenceEnegiesFile + " &> " +  paramfitOutFilename  )
        
        K = -1
        paramfitOut = open( paramfitOutFilename, 'r' )
        line = paramfitOut.readline()
        while line:
            if "*K = " in line:
                K = line.split()[2]
            line =paramfitOut.readline()
        
        paramfitOut.close()
        
        mmEnergiesLog = join(self.scratchDir, "paramfitEnergies.log")
        
        paramfitInputFilename = join(self.scratchDir, "paramfit2.in") 
        paramfitInput = open( paramfitInputFilename , 'w' )
        paramfitInput.write( """RUNTYPE=FIT
NSTRUCTURES={strNo}
COORDINATE_FORMAT=TRAJECTORY
ALGORITHM=NONE
FUNC_TO_FIT=SUM_SQUARES_AMBER_STANDARD
K={K}
WRITE_ENERGY={MMenergiesFile}

""".format( strNo = strNo, K = K, MMenergiesFile = mmEnergiesLog ) )
        paramfitInput.close()
        
        paramfitOutFilename = join(self.scratchDir, "paramfit2.out")
        system( "paramfit -i "+paramfitInputFilename+" -p " + self.currentPrmtop + " -c " + self.referenceTrajectory + " -q " + self.referenceEnegiesFile + " &> " +  paramfitOutFilename  )
        
        return np.loadtxt( mmEnergiesLog, skiprows = 2 )
    
    def runEvaluationMD(self):
        minimizedCoordsFile = join(self.scratchDir, "amberMinimized.nc")
        minimizationTraj = join(self.scratchDir, "amberMinTraj.nc")
        minimizationOut = join(self.scratchDir, "amberMin.log")

        mdOut = join(self.scratchDir, "amberMD.log")
        mdRest =  join(self.scratchDir, "amberMD_rest.nc")
        system("$AMBERHOME/bin/sander -O -i " + self.sanderMinFile + " -c " + self.mdSeed + " -p " + self.currentPrmtop  + "  -o "+minimizationOut+" -r " +minimizedCoordsFile+ " -x "+minimizationTraj  )
        system("$AMBERHOME/bin/sander -O -i " + self.sanderMDFile + " -c " + minimizedCoordsFile + " -p " + self.currentPrmtop  + "  -o "+mdOut+" -r "+mdRest+" -x "+self.evaluationMDtrajectory  )
    
#        system("mpirun -np 24 $AMBERHOME/bin/sander.MPI -O -i " + self.sanderMinFile + " -c " + self.mdSeed + " -p " + self.currentPrmtop  + "  -o "+minimizationOut+" -r " +minimizedCoordsFile+ " -x "+minimizationTraj  )
#        system("mpirun -np 24 $AMBERHOME/bin/sander.MPI -O -i " + self.sanderMDFile + " -c " + minimizedCoordsFile + " -p " + self.currentPrmtop  + "  -o "+mdOut+" -r "+mdRest+" -x "+self.evaluationMDtrajectory  )
  
    
    def evaluateMD(self):
        return self.evaluateWithCppTraj( self.evaluationMDtrajectory )
    
    def goalFunction(self, X):
        energyWeight = 25**2
        bondWeight = 100**2
        angleWeight = 2**2
        
        goalFuntionValue = 0
        
        print("Obliczanie funkcji celu: start")
        print("Wektor X: ", X)
        self.cleanScratch()
        self.setX(X)
        
        energiesFromParamfit = self.evaluateWithParamFit()
        
        amberEnergies = energiesFromParamfit[:,1]
        qmEnergies = energiesFromParamfit[:,2]
        
        meanEdiffSquare = np.mean( np.square( amberEnergies - qmEnergies ) )
        print("Srednia kwadratow roznic energii: ", meanEdiffSquare)
        goalFuntionValue += energyWeight*meanEdiffSquare
        print("Wklad do funkcji celu: ", goalFuntionValue)
        
        self.runEvaluationMD()
        geometricalMeans = self.evaluateMD()
        
        for evalParm in geometricalMeans:
            referenceValue = self.referenceMeans[evalParm]
            currentValue = geometricalMeans[evalParm]
            parmKind = self.evaluationKinds[evalParm]
            
            diffSquare = (referenceValue-currentValue)**2
            print("Kwadrat roznicy srednich parametru geometrycznego: ", diffSquare, evalParm)
            
            if parmKind == "distance":
                goalFuntionValue += bondWeight*diffSquare
                print("Wklad do funkcji celu: ", bondWeight*diffSquare)
            elif parmKind == "angle":
                goalFuntionValue += angleWeight*diffSquare
                print("Wklad do funkcji celu: ", angleWeight*diffSquare)
                
        print("Wartosc funkcji celu: ", goalFuntionValue)

        return goalFuntionValue
        
fitter = ParameterOptimizer( prmtopInit = "example/modelInit.prmtop", 
                        referenceTrajectory = "example/referenceTrajectory.mdcrd", 
                        referenceTrajectoryForCppTraj="example/referenceTrajectory.dcd",
                        evaluationCriteria = "example/geometricCriteria.ptraj", 
                        referenceEnergies = "example/referenceEnergies.dat",
                        sanderMinimization= "example/sander_min.in",
                        sanderMD= "example/sander_md.in",
                        MDseed= "example/mdSeed.prmcrd",
                        parameters2optimizeFile = "example/params2optimize.dat")
initialX = fitter.readX()
fitter.goalFunction(initialX)
res = minimize(fitter.goalFunction, initialX, method='nelder-mead', bounds = fitter.getBounds(),
               options={'xatol': 300, 'disp': True})
