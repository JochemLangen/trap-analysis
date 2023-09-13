# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 12:50:47 2022

@author: joche
"""

from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
from OpticsFuncs import *


#Set initial beam parameters:
GridSize = 20*mm
GridDimension = 2000
lambda_ = 532*nm #lambda_ is used because lambda is a Python build-in function.
w0 = 2*mm #Gaussian waist size
z_R = (np.pi*w0**2)/lambda_
print("Rayleigh length of the Gaussian beam is:{} mm".format(z_R/mm))
del_z_laser_axicon1 = 10*mm
axicon1_phi = 175/180*np.pi
axicon1_n = 1.5
del_z_axicon1_lens1 = 100*mm
lens1_f = 100*mm
del_z_lens1_axicon2 = 300*mm
axicon2_phi = 5/180*np.pi
z=0

#Initialise reference field of intensity 1
Field = Begin(GridSize, lambda_, GridDimension)
# print("Initial field:", Field.field)

#Initialise Gaussian beam (Hermite) at waist
Field = GaussBeam(Field, w0)
xy_Intensity_plotter(Field, GridDimension, GridSize,"Initial Gaussian beam at z = {} m".format(round(z,3)))
print("Initial Gaussian:",Field.field)

#Propagates field using a convolution method:
steps = 1
for i in range(steps):
    z += del_z_laser_axicon1/steps
    Field = Forvard(Field, del_z_laser_axicon1/steps)
    xy_Intensity_plotter(Field, GridDimension, GridSize, "Propagated Gaussian beam at z = {} m".format(round(z,3)))
    
#Calculate beam width
w_ = w0*np.sqrt(1+(z/z_R)**2) #Laser beam width if z >> z_R

w = beam_width(Field, GridDimension, GridSize)

#Calculate the predicted distance between axicon and end of bessel wave components
Z_max = w/(axicon1_n-1)/(np.pi-axicon1_phi) #The distance from the axicon from which the hole forms
Z_max_err = abs(Z_max-(w+GridSize/GridDimension)/(axicon1_n-1)/(np.pi-axicon1_phi))

#Propagate field through axicon:    
Field = Axicon(Field, axicon1_phi, axicon1_n, x_shift=0.0, y_shift=0.0)
xy_Intensity_plotter(Field, GridDimension, GridSize, "Gaussian beam through axicon at z = {} m".format(round(z,3)))
# print("Initial Gaussian:",Field.field)

print("Z_max is: {} +/- {} mm".format(Z_max/mm,Z_max_err/mm))

#Propagates field using a FFT method:
steps = 3
for i in range(steps):
    z += del_z_axicon1_lens1/steps
    Field = Forvard(Field, del_z_axicon1_lens1/steps)
    xy_Intensity_plotter(Field, GridDimension, GridSize, "Propagated Bessel beam at z = {} m".format(round(z,3)))
    
#Propagate field through lens:
Field = Lens(Field,lens1_f)



#Propagates field using a FFT method:
steps = 10
Field, z = waterfall_plotter_lines(Field, GridDimension, GridSize, "Propagated Hollow beam", del_z_lens1_axicon2, steps, z)

# for i in range(steps):
#     z += del_z_lens1_axicon2/steps
#     Field = Forvard(Field, del_z_lens1_axicon2/steps)
#     xy_Intensity_plotter(Field, GridDimension, GridSize, "Propagated Hollow beam at z = {} m".format(round(z,3)))

#Propagate field through axicon:
# Field = Axicon(Field, axicon2_phi, axicon1_n, x_shift=0.0, y_shift=0.0)

# #Propagates field using a FFT method:
# steps = 6
# for i in range(steps):
#     z += del_z_axicon1_lens1/steps/10
#     Field = Forvard(Field, del_z_axicon1_lens1/steps)
#     xy_Intensity_plotter(Field, GridDimension, GridSize, "Propagated Hollow beam at z = {} m".format(round(z,3)))