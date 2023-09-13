# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 15:40:42 2022

@author: joche
"""

from LightPipes import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


wavelength=532*nm
size=10*mm
N=1000

N2=int(N/2)
ZoomFactor=1
NZ=N2/ZoomFactor

phi=175/180*3.1415; n1=1.5
z_start=0.001*cm; z_end= 24*cm;
steps=4;
delta_z=(z_end-z_start)/steps
z=z_start

F=Begin(size,wavelength,N);
F=GaussBeam(F, size/5)
F=Axicon(F,phi,n1,0,0)

for i in range(1,steps): 
    z=z+delta_z
    F=Forvard(delta_z,F);
    I=Intensity(0,F);
    plt.subplot(2,5,i)
    s='z= %3.3f m' % (z/m)
    plt.title(s)
    image = plt.imshow(I,cmap='rainbow', vmax=1);plt.axis('off')
    # plt.colorbar(mappable=image,label='Normalised intensity')
    plt.axis([N2-NZ, N2+NZ, N2-NZ, N2+NZ])
    

plt.show()