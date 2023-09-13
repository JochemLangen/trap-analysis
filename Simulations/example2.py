# -*- coding: utf-8 -*-
"""
Created on Tue Nov  1 23:23:57 2022

@author: joche
"""
from LightPipes import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from LightpipesFuncs import *

wavelength=1000.0*nm
size=10*mm
N=2000

N2=int(N/2)
ZoomFactor=1
NZ=N2/ZoomFactor

phi=179.7/180*3.1415; n1=1.5
z_start=0.001*cm; z_end= 200*cm;
steps=11;
delta_z=(z_end-z_start)/steps
z=z_start

F=Begin(size,wavelength,N);
F=GaussBeam(F, size/5)
yy, xx = F.mgrid_cartesian
print(xx[0],xx[:,0])
print(xx,yy)
F=Axicon(F,phi,n1,0,0)


for i in range(1,steps): 
    F=Forvard(delta_z,F);
    I=Intensity(0,F);
    plt.subplot(2,5,i)
    s='z= %3.1f m' % (z/m)
    plt.title(s)
    plt.imshow(I,cmap='plasma', vmax=1);plt.axis('off')
    plt.axis([N2-NZ, N2+NZ, N2-NZ, N2+NZ])
    z=z+delta_z
plt.show()