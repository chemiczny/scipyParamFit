#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 11:43:56 2020

@author: michal
"""

from scipy.optimize import minimize
from scipyParamFit import ParameterOptimizer
from parmed.amber import AmberMdcrd

def xyz2mdcrd( xyz, mdcrd ):
    xyzF = open(xyz, 'r')
    line = xyzF.readline()
    atomsNo = int(line)
    mdcrdTraj = AmberMdcrd( mdcrd, atomsNo, hasbox = False, mode = 'w' )
    
    while line:
        
        coordsFrame = []
        xyzF.readline()
        for i in range(atomsNo):
            line = xyzF.readline()
            for c in line.split()[-3:]:
                coordsFrame.append(float(c) )
        
        mdcrdTraj.add_coordinates(coordsFrame)
        line = xyzF.readline()
    
    xyzF.close()
    
    mdcrdTraj.close()

xyzRef = "example/referenceTrajectory.xyz"
mdcrdRef = "referenceTrajectory.crd"
xyz2mdcrd(xyzRef, mdcrdRef)

fitter = ParameterOptimizer( prmtopInit = "example/modelInit.prmtop", 
                        referenceTrajectory = mdcrdRef, 
                        evaluationCriteria = "example/geometricCriteria.ptraj", 
                        referenceEnergies = "example/referenceEnergies.dat",
                        sanderMinimization= "example/sander_min.in",
                        sanderMD= "example/sander_md.in",
                        MDseed= "example/mdSeed.prmcrd",
                        parameters2optimizeFile = "example/params2optimize.dat")
initialX = fitter.readX()
#fitter.goalFunction(initialX)
res = minimize(fitter.goalFunction, initialX, method='nelder-mead', bounds = fitter.getBounds(),
               options={'xatol': 300, 'disp': True})