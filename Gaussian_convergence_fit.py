# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 15:24:07 2022

@author: joche
"""

from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
from OpticsFuncs import *

#Set initial beam parameters:
GridSize = 6*mm
GridDimension = 2500
lambda_ = 500*nm #lambda_ is used because lambda is a Python build-in function.
w0 = 0.2*mm #Gaussian waist size
z_R1 = (np.pi*w0**2)/lambda_
print("Rayleigh length of the initial Gaussian beam is: {} mm".format(z_R1/mm))
print("Boundary of simulation area is: {}x w0".format(GridSize/2/w0))
initial_z=0
final_z = 0.5*m
lens1_f = 0.05*m
z_R2 = z_R1/(z_R1 + (z_R1/lens1_f)**2) #Note this only works for a gaussian beam with its waist ~ in the lens plane
print("Rayleigh length of the focussed Gaussian beam is (z_R2): {} mm".format(z_R2/mm))

z0 = lens1_f-z_R2**2/lens1_f
print("Distance to focussed beam waist: {} mm".format(z0/mm))

wf = np.sqrt(z_R2*lambda_/np.pi)
print("Waist radius of the focussed beam (w_f): {} mm".format(wf/mm))

#Initialise reference field of intensity 1
Field = Begin(GridSize, lambda_, GridDimension)
# print("Initial field:", Field.field)

#Initialise Gaussian beam (Hermite) at waist
Field = GaussBeam(Field, w0)

w = beam_width(Field, GridDimension, GridSize)
print(w)

w_unc = GridSize/GridDimension
steps = 20
w_list = np.empty(steps)
w_list[0] = w
step_z = final_z/steps
for i in range(1,steps):
    # Field = Forvard(Field, step_z)
    Field = Steps(Field, step_z)
    w_list[i] = beam_width(Field, GridDimension, GridSize)

z_list = np.linspace(0,final_z,steps)/z_R1
w_list /= w0
w_unc /= w0

chi_sqrd = np.sum((w_list-norm_gaussian_width(z_list))**2)/len(z_list)
print("Reduced chi squared is: {}".format(chi_sqrd))
z_points = np.linspace(0,final_z,1000)/z_R1
w_func = norm_gaussian_width(z_points)


plt.figure()
plt.errorbar(z_list,w_list,yerr=w_unc)
plt.plot(z_points,w_func)
plt.xlabel(r"Normalised distance from beam waist ($z/z_{R1}$)")
plt.ylabel(r"Normalised beam waist ($w/w_{0}$)")
plt.show()


# #Propagate field through lens:
# Field = Lens(Field,lens1_f)


# w_unc = GridSize/GridDimension
# steps = 20
# w_list = np.empty(steps)
# w_list[0] = w
# step_z = final_z/steps
# for i in range(1,steps):
#     Field = Forvard(Field, step_z)
#     w_list[i] = beam_width(Field, GridDimension, GridSize)

# z_list = (np.linspace(0,final_z,steps)-z0)/z_R2
# w_list /= wf
# w_unc /= wf

# def norm_gaussian_width(norm_z):
#     return np.sqrt(1+(norm_z)**2)

# z_points = (np.linspace(0,final_z,1000)-z0)/z_R2
# w_func = norm_gaussian_width(z_points)


# plt.figure()
# plt.errorbar(z_list,w_list,yerr=w_unc)
# plt.plot(z_points,w_func)
# plt.xlabel(r"Normalised distance from beam waist ($z/z_{R2}$)")
# plt.ylabel(r"Normalised beam waist ($w/w_{f}$)")
# plt.show()
# # Y = z - z_lens - lens1_f
# print('Focal distance to waist {} +/- {}'.format(Y,step_z))
# print('Factor between space before and after lens: {} % +/- {} %'.format(Y/del_z,step_z/del_z))