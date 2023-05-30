# -*- coding: utf-8 -*-
"""
Created on Thu Oct 27 22:01:19 2022

@author: joche
"""

from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
from OpticsFuncs import *

#Set initial beam parameters:
GridSize = 6*mm
GridDimension = 2000
lambda_ = 500*nm #lambda_ is used because lambda is a Python build-in function.
w0 = 0.2*mm #Gaussian waist size
z_R = (np.pi*w0**2)/lambda_
print("Rayleigh length of the Gaussian beam is:{} mm".format(z_R/mm))
del_z = 0.164*m
z=0
lens1_f = 0.6*m
step_z = 0.01*m

#Initialise reference field of intensity 1
Field = Begin(GridSize, lambda_, GridDimension)
# print("Initial field:", Field.field)

#Initialise Gaussian beam (Hermite) at waist
Field = GaussBeam(Field, w0)

w = beam_width(Field, GridDimension, GridSize)
print(w)

Field = Forvard(Field, del_z+lens1_f)
z += del_z+lens1_f

#Propagate field through lens:
Field = Lens(Field,lens1_f)

z_lens = z

Field = Forvard(Field, lens1_f)
z += lens1_f


# w_old = beam_width(Field, GridDimension, GridSize)
# print(w_old)
# print("start loop")
# # w_unc = GridSize/GridDimension
# counter = 0
# for i in range(200):
#     Field = Forvard(Field, step_z)
#     w_new = beam_width(Field, GridDimension, GridSize)
#     print(w_new)
#     if w_new == w_old:
#         counter += 1
#     elif w_new < w_old:
#         counter = 0
#     if w_new > w_old:
#         z += step_z*(i-1-int(counter/2))
#         print(counter, int(counter/2))
#         print(w_old)
#         print("index:", i)
#         break
#     w_old = w_new

# Y = z - z_lens - lens1_f
# print('Focal distance to waist {} +/- {}'.format(Y,step_z))
# print('Factor between space before and after lens: {} % +/- {} %'.format(Y/del_z,step_z/del_z))

steps = 100
Field, z, del_z_lens_waist = waterfall_plotter_gaussian_test(Field, GridDimension, GridSize, "Gaussian beam", 1, steps, z,zoomfactor = 2)
print('Distance between focal point and waist: {} +/- {} m'.format(del_z_lens_waist,step_z))
print('Factor between space before and after lens: {} +/- {}'.format(del_z_lens_waist/del_z,step_z/del_z))