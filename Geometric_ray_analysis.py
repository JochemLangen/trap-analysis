# -*- coding: utf-8 -*-
"""
Created on Thu Feb 23 14:18:07 2023

@author: joche


Basic geometric ray tracing calculation:

System description:
1. Coupled optical fibre (no collimating lens)
2. Collimating lens of 25 mm in diameter
3. 1st telescope lens
4. 2nd telescope lens
    
All units are in mm

Variable (to be determined in the lab):
-Divergence of Gaussian beam out of fibre tip.
-> Z_R = pi*w_0^2/lambda
-> w = w_0*(1+(z/z^R)^2)^1/2
-> z = ((w/w_0)^2-1)^1/2 * z_R
"""

import numpy as np
pi = np.pi
wl = 532*(10**-6) #mm


def tn_lens(f,s_object):
    s_image = 1/(1/f-1/s_object)
    m = -s_image/s_object
    return s_image, m

def tn_lens_gauss(f,z_R1,z1,wl):
    denom = (1-z1/f)**2 + (z_R1/f)**2
    z2 = (z1 - (z1**2)/f - (z_R1/f)**2)/denom #Positive is upstream from the lens
    z_R2 = z_R1/denom
    w_2 = np.sqrt(wl*z_R2/pi)
    return z2, z_R2, w_2

def Rayleigh_ln(w_0):
    return (pi*w_0**2)/wl

#Modefield diameter fibre
w_0 = 0.004
z_R0 = Rayleigh_ln(w_0)
w_lens1 = 12.5 / 6 
z_0 = np.sqrt((w_lens1/w_0)**2 - 1)*z_R0
print(z_0)

f_lens1 = 40 
z1, z_R1, w_1 = tn_lens_gauss(f_lens1,z_R0,z_0,wl)
print("Length from lens 1 to focus {} mm, with waist size {} mm".format(-z1,w_1))

w_lens2 = 12.5/3
f_lens2 = 75
d_lens1_lens2 = np.sqrt((w_lens2/w_1)**2 - 1)*z_R1 - z1
print("Required distance between lens 1 & lens 2: {} mm".format(d_lens1_lens2))
z2,z_R2, w_2 = tn_lens_gauss(f_lens2,z_R1,z1+d_lens1_lens2,wl)
print("Length from lens 2 to focus {} mm, with waist size {} mm".format(-z2,w_2))


w_lens3 = 12.5 / 3
f_lens3 = 300
d_lens2_lens3 = np.sqrt((w_lens3/w_2)**2-1)*z_R2-z2
print("Required distance between lens 2 & lens 3: {} mm".format(d_lens2_lens3))
z3,z_R3,w_3 = tn_lens_gauss(f_lens3,z_R2,z2+d_lens2_lens3,wl)
print("Length from lens 3 to focus {} mm, with waist size {} mm".format(-z3,w_3))

d_fbr_lens1 = 25 #Distance from optical fibre to the first lens
# w_0 = w_lens1/(1+d_fbr_lens**2/`z_r)
#Z_r
s_obj_lens1 = d_fbr_lens1


s_img_lens1, m_lens1 = tn_lens(f_lens1,s_obj_lens1)
print("Image distance")
d_lens1_lens2 = 50
s_obj_lens2 = d_lens1_lens2 - s_img_lens1
f_lens2 = 100

s_img_lens2, m_lens2 = tn_lens(f_lens2,s_obj_lens2)
m_total = m_lens1*m_lens2
print(s_img_lens2,m_total)

total_s_img = s_img_lens2 + d_lens1_lens2
f_eff = 1/(1/s_obj_lens1+1/total_s_img)
print(f_eff)