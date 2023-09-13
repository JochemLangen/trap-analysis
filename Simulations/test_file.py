# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 14:59:28 2022

@author: joche
"""

from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
from OpticsFuncs import *
from LightpipesFuncs import *


#Set initial beam parameters:
GridSize = 10*mm
GridDimension = 3000
lambda_ = 532*nm #lambda_ is used because lambda is a Python build-in function.
w0 = 1.5*mm #Gaussian waist size
z_R = (np.pi*w0**2)/lambda_
print("Rayleigh length of the Gaussian beam is:{} mm".format(z_R/mm))

z=0
axicon1_phi = 175/180*np.pi
axicon2_phi = 170/180*np.pi
axicon1_n = 1.46
Axicon_tip_R = 1*mm
Axicon_edge_width = 1*mm
Axicon_radius = 12.7*mm
print
#Initialise reference field of intensity 1
Field = Begin(GridSize, lambda_, GridDimension)
# print("Initial field:", Field.field)

#Initialise Gaussian beam (Hermite) at waist
Field = GaussBeam(Field, w0)
xy_Intensity_plotter(Field, GridDimension, GridSize,"Initial Gaussian beam at z = {} m".format(round(z,3)))

# #Propagates field using a FFT method:
# steps = 10
# Field, z = waterfall_plotter_lines(Field, GridDimension, GridSize, "Gaussian beam", del_z, steps, z,zoomfactor = 1.5)

#Propagate field through axicon:    
Field = real_Axicon(Field, axicon1_phi, n1 = axicon1_n, R = Axicon_tip_R, d = Axicon_edge_width, w=Axicon_radius, x_shift = 0.0, y_shift = 0.0, geo_adjust=False)
# Field = Axicon(Field, axicon1_phi, axicon1_n, x_shift=0.0, y_shift=0.0)
print(Field._curvature)
print(Field._q)
# xy_Intensity_plotter(Field, GridDimension, GridSize,"Beam straight after Axicon at z = {} m".format(round(z,3)))

# w = beam_width(Field, GridDimension, GridSize)

# #Calculate the predicted distance between axicon and end of bessel wave components
# Z_max = w/(axicon1_n-1)/(np.pi-axicon1_phi) #The distance from the axicon from which the hole forms
# Z_max_err = abs(Z_max-(w+GridSize/GridDimension)/(axicon1_n-1)/(np.pi-axicon1_phi))
# print("Z_max is: {} +/- {} mm".format(Z_max/mm,Z_max_err/mm))

# #Propagates field using a FFT method:
# # jump_z = 85*mm
# # Field = Forvard(Field, jump_z)
# # z += jump_z    

steps = 2
del_z = 90*mm
Field = Forvard(Field, del_z)
z += del_z 
# Field, z = waterfall_plotter_lines(Field, GridDimension, GridSize, "Propagated beam after axicon", del_z, steps, z, zoomfactor=1)
xy_Intensity_plotter(Field, GridDimension, GridSize,"Initial Gaussian beam at z = {} m".format(round(z,3)))

#Propagate field through lens:
Field = Lens(Field,150*mm)

steps = 2
del_z = 20*mm
Field = Forvard(Field, del_z)
z += del_z 
# Field, z = waterfall_plotter_lines(Field, GridDimension, GridSize, "Propagated beam after axicon", del_z, steps, z, zoomfactor=1)
xy_Intensity_plotter(Field, GridDimension, GridSize,"Initial Gaussian beam at z = {} m".format(round(z,3)))


#Propagate field through axicon:    
Field = real_Axicon(Field, axicon2_phi, n1 = axicon1_n, R = Axicon_tip_R, d = Axicon_edge_width, w=Axicon_radius, x_shift = 0.0, y_shift = 0.0, geo_adjust=False)


steps = 2
del_z = 50*mm
Field = Forvard(Field, del_z)
z += del_z 
# Field, z = waterfall_plotter_lines(Field, GridDimension, GridSize, "Propagated beam after axicon", del_z, steps, z, zoomfactor=1)
xy_Intensity_plotter(Field, GridDimension, GridSize,"Initial Gaussian beam at z = {} m".format(round(z,3)))

del_z = 50*mm
Field = Forvard(Field, del_z)
z += del_z 
# Field, z = waterfall_plotter_lines(Field, GridDimension, GridSize, "Propagated beam after axicon", del_z, steps, z, zoomfactor=1)
xy_Intensity_plotter(Field, GridDimension, GridSize,"Initial Gaussian beam at z = {} m".format(round(z,3)))


#Propagate field through axicon:    
Field = real_Axicon(Field, axicon1_phi, n1 = axicon1_n, R = Axicon_tip_R, d = Axicon_edge_width, w=Axicon_radius, x_shift = 0.0, y_shift = 0.0, geo_adjust=False)

steps = 2
del_z = 30*mm
Field = Forvard(Field, del_z)
z += del_z 
# Field, z = waterfall_plotter_lines(Field, GridDimension, GridSize, "Propagated beam after axicon", del_z, steps, z, zoomfactor=1)
xy_Intensity_plotter(Field, GridDimension, GridSize,"Initial Gaussian beam at z = {} m".format(round(z,3)))
