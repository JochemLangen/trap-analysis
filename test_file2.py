# -*- coding: utf-8 -*-
"""
Created on Thu Nov  3 11:52:17 2022

@author: joche
"""

from LightPipes import *
import matplotlib.pyplot as plt
import numpy as np
# from OpticsFuncs import *
from LightpipesFuncs import plot_real_Axicon

#Set initial beam parameters:
GridSize = 15*mm
GridDimension = 2000
lambda_ = 532*nm #lambda_ is used because lambda is a Python build-in function.
w0 = 2*mm #Gaussian waist size
z_R = (np.pi*w0**2)/lambda_
print("Rayleigh length of the Gaussian beam is:{} mm".format(z_R/mm))
del_z = 200*mm
z=0
axicon1_phi = 175/180*np.pi
axicon1_n = 1.5
Axicon_tip_R = 20*mm
Axicon_edge_width = 0*mm
Axicon_radius = 20*mm

#Initialise reference field of intensity 1
Field = Begin(GridSize, lambda_, GridDimension)


plot_real_Axicon(Field, axicon1_phi, n1 = axicon1_n, R = Axicon_tip_R, d = Axicon_edge_width, w=Axicon_radius, x_shift = 0.0, y_shift = 0*mm,n_lines=40)


